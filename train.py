import tensorflow as tf
from glob import glob
from deeplab_test import DeepLabV3Plus
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import os
import math
from lovasz_softmax import lovasz_hinge
print('TensorFlow', tf.__version__)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


H, W = 240, 320
batch_size = 15

train_images = sorted(glob('/home/igs/Documents/soiling_dataset-all/train/rgbImages/*'))
train_masks = sorted(glob('/home/igs/Documents/soiling_dataset-all/train/binaryLabels/*'))

val_images = sorted(glob('/home/igs/Documents/soiling_dataset-all/test/rgbImages/*'))
val_masks = sorted(glob('/home/igs/Documents/soiling_dataset-all/test/binaryLabels/*'))

print(f'Found {len(train_images)} training images')
print(f'Found {len(train_masks)} training masks')

print(f'Found {len(val_images)} validation images')
print(f'Found {len(val_masks)} validation masks')

for i in range(len(train_masks)):
    assert train_images[i].split(
        '/')[-1].split('.')[0] == train_masks[i].split('/')[-1].split('.')[0]

for i in range(len(val_masks)):
    assert val_images[i].split(
        '/')[-1].split('.')[0] == val_masks[i].split('/')[-1].split('.')[0]


def random_scale(image, mask, min_scale=0.65, max_scale=2.5):
    random_scale = tf.random.uniform(shape=[1],
                                     minval=min_scale,
                                     maxval=max_scale)
    dims = tf.cast(tf.shape(image), dtype=tf.float32)
    new_dims = tf.cast(random_scale * dims[:2], dtype=tf.int32)
    scaled_image = tf.image.resize(image, size=new_dims, method='bilinear')
    scaled_mask = tf.image.resize(mask, size=new_dims, method='nearest')
    return scaled_image, scaled_mask


def pad_inputs(image,
               mask,
               crop_height=H,
               crop_width=W,
               ignore_value=255,
               pad_value=0):
    dims = tf.cast(tf.shape(image), dtype=tf.float32)
    h_pad = tf.maximum(1 + crop_height - dims[0], 0)
    w_pad = tf.maximum(1 + crop_width - dims[1], 0)
    padded_image = tf.pad(image, paddings=[[0, h_pad], [0, w_pad], [
                          0, 0]], constant_values=pad_value)
    padded_mask = tf.pad(mask, paddings=[[0, h_pad], [0, w_pad], [
                         0, 0]], mode='CONSTANT', constant_values=ignore_value)
    return padded_image, padded_mask


def random_crop(image, mask, crop_height=H, crop_width=W):
    image_dims = tf.shape(image)
    offset_h = tf.random.uniform(
        shape=(1,), maxval=image_dims[0] - crop_height, dtype=tf.int32)[0]
    offset_w = tf.random.uniform(
        shape=(1,), maxval=image_dims[1] - crop_width, dtype=tf.int32)[0]

    image = tf.image.crop_to_bounding_box(image,
                                          offset_height=offset_h,
                                          offset_width=offset_w,
                                          target_height=crop_height,
                                          target_width=crop_width)
    mask = tf.image.crop_to_bounding_box(mask,
                                         offset_height=offset_h,
                                         offset_width=offset_w,
                                         target_height=crop_height,
                                         target_width=crop_width)
    return image, mask


def random_flip(image, mask):
    flip = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    image = tf.case([
        (tf.greater(flip, 0), lambda: tf.image.flip_left_right(image))
    ], default=lambda: image)
    mask = tf.case([
        (tf.greater(flip, 0), lambda: tf.image.flip_left_right(mask))
    ], default=lambda: mask)
    return image, mask


def load_image(image_path, mask=False):
    img = tf.io.read_file(image_path)
    if mask:
        img = tf.image.decode_image(img, channels=1)
        img.set_shape([None, None, 1])
    else:
        img = tf.image.decode_image(img, channels=3)
        img.set_shape([None, None, 3])
    return img


@tf.function()
def preprocess_inputs(image_path, mask_path):
    with tf.device('/cpu:0'):
        image = load_image(image_path)
        mask = load_image(mask_path, mask=True)
        mask = tf.cast(mask > 0, dtype=tf.uint8)

        image, mask = random_scale(image, mask)
        image, mask = pad_inputs(image, mask)
        image, mask = random_crop(image, mask)
        image, mask = random_flip(image, mask)
        image = image[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
        return image, mask


train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
train_dataset = train_dataset.shuffle(1024)
train_dataset = train_dataset.map(map_func=preprocess_inputs,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
val_dataset = val_dataset.shuffle(512)
val_dataset = val_dataset.map(map_func=preprocess_inputs,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size=batch_size, drop_remainder=True)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

print(train_dataset)


@tf.function()
def dice_coef(y_true, y_pred):
    mask = tf.equal(y_true, 255)
    mask = tf.logical_not(mask)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


@tf.function()
def bce_loss(y_true, y_pred):
    mask = tf.equal(y_true, 255)
    mask = tf.logical_not(mask)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return tf.losses.binary_crossentropy(y_true, y_pred)

@tf.function()    
def lovaszsoftmax(y_true, y_pred):
    mask = tf.equal(y_true, 255)
    mask = tf.logical_not(mask)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return lovasz_hinge(y_pred,y_true)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = DeepLabV3Plus(H, W)
    #TODO: Regularization loss model.add_loss(regularizer(model.layers[i].kernel))
    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss=lovaszsoftmax,
    #              optimizer=tf.keras.optimizers.Adam(2e-5),
    #              metrics=['accuracy', dice_coef])

    model.compile(loss=lovaszsoftmax,
                  optimizer=sgd,
                  metrics=['accuracy', dice_coef])



tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
mc = ModelCheckpoint(filepath='models/model-ep{epoch:03d}-loss{loss:.3f}.h5',
                     monitor='val_loss',
                     mode='auto',
                     save_best_only='True',
                     save_weights_only='True', verbose=1)


def learning_rate_fn(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
    #if epoch < 50:
    #    return 1e-4
    #elif epoch < 100:
    #    return 2e-4
    #elif epoch <= 150:
    #    return 1e-4
    #elif epoch > 200:
    #    return 5e-5


lr_schedule = tf.keras.callbacks.LearningRateScheduler(learning_rate_fn)

callbacks = [mc, tb, lr_schedule]


model.fit(train_dataset,
          steps_per_epoch=len(train_images) // batch_size,
          epochs=300,
          validation_data=val_dataset,
          validation_steps=len(val_images) // batch_size,
          callbacks=callbacks)
model.save_weights('last_epoch.h5')
