#! -*- coding: utf-8 -*-
import numpy as np
from scipy import misc
import glob
import imageio
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from tensorflow.python.ops import image_ops
#os.environ['CUDA_VISIBLE_DEVICES']=''

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


imgs = glob.glob('/home/igs/Documents/soiling_dataset-all/train/binaryLabels/*.png')
np.random.shuffle(imgs)

height,width = misc.imread(imgs[0]).shape[:2]
center_height = int((height - width) / 2)
img_dim_h = 240
img_dim_w = 320
z_dim = 512


def imread(f):
    x = misc.imread(f)
    x = misc.imresize(x, (img_dim_h, img_dim_w))
    return x.astype(np.float32) / 255


def data_generator(batch_size=96):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X,None
                X = []


x_in = Input(shape=(img_dim_h, img_dim_w, 3))
x = x_in
x = Conv2D(z_dim/16, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim/8, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim/4, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim/2, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = GlobalAveragePooling2D()(x)

encoder = Model(x_in, x)
encoder.summary()
map_size = K.int_shape(encoder.layers[-2].output)[1:-1]

z_in = Input(shape=K.int_shape(x)[1:])
z = z_in
z = Dense(np.prod(map_size)*z_dim)(z)
z = Reshape(map_size + (z_dim,))(z)
z = Conv2DTranspose(z_dim/2, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(z_dim/4, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(z_dim/8, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(z_dim/16, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(3, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = Activation('sigmoid')(z)
z = image_ops.resize_images_v2(z,[img_dim_h,img_dim_w], method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)
decoder = Model(z_in, z)
decoder.summary()

class ScaleShift(Layer):
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def call(self, inputs):
        z, shift, log_scale = inputs
        z = K.exp(log_scale) * z + shift
        logdet = -K.sum(K.mean(log_scale, 0))
        self.add_loss(logdet)
        return z

z_shift = Dense(z_dim)(x)
z_log_scale = Dense(z_dim)(x)
u = Lambda(lambda z: K.random_normal(shape=K.shape(z)))(z_shift)
z = ScaleShift()([u, z_shift, z_log_scale])

x_recon = decoder(z)
x_out = Subtract()([x_in, x_recon])

#recon_loss = 0.5 * K.sum(K.mean(x_out**2, 0)) + 0.5 * np.log(2*np.pi) * np.prod(K.int_shape(x_out)[1:])
recon_loss = K.mean(K.sum(K.binary_crossentropy(x_in,x_recon),[1,2,3]),axis=0)
z_loss = 0.5 * K.sum(K.mean(z**2, 0)) - 0.5 * K.sum(K.mean(u**2, 0))
vae_loss = 0.01* recon_loss + z_loss

vae = Model(x_in, x_out)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(1e-4))


def sample(path):
    n = 3
    figure = np.zeros((img_dim_h*n, img_dim_w*n, 3))
    for i in range(n):
        for j in range(n):
            #x_recon = decoder.predict(np.random.randn(1, *K.int_shape(x)[1:]))
            x_recon = decoder.predict(np.random.randn(1, *(z_dim,)))
            digit = x_recon[0]
            figure[i*img_dim_h: (i+1)*img_dim_h,
                   j*img_dim_w: (j+1)*img_dim_w] = digit
    figure = figure* 255
    imageio.imwrite(path, figure)


class Evaluate(Callback):
    def __init__(self):
        import os
        self.lowest = 1e10
        self.losses = []
        if not os.path.exists('Ori_samples'):
            os.mkdir('Ori_samples')
    def on_epoch_end(self, epoch, logs=None):
        path = 'Ori_samples/test_%s.png' % epoch
        sample(path)
        self.losses.append((epoch, logs['loss']))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('Ori_model/best_encoder.weights')
            decoder.save_weights('Ori_model/best_decoder.weights')
            vae.save('Ori_model/best_model.h5')

evaluator = Evaluate()
finetune = False
if finetune:
    encoder.load_weights('Ori_model/best_encoder.weights')
    decoder.load_weights('Ori_model/best_decoder.weights')

vae.fit_generator(data_generator(),
                  epochs=2000,
                  steps_per_epoch=100,
                  callbacks=[evaluator])