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
#from self_layer import ReflectionPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import os
import cv2
#os.environ['CUDA_VISIBLE_DEVICES']=''

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

    def get_config(self):  
        config = {"self.padding": self.padding,"self.input_spec": self.input_spec}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





imgs = glob.glob('/home/igs/Documents/soiling_dataset-all/train/binaryLabels/*.png')
#imgs = glob.glob('/home/igs/Documents/vae/test_data/*.png')
np.random.shuffle(imgs)

height,width = misc.imread(imgs[0]).shape[:2]
img_dim_h = 240
img_dim_w = 320
z_dim = 100
finetune = True

def imread(f):
    x = misc.imread(f)
    x = misc.imresize(x[:,:,1], (img_dim_h, img_dim_w))
    x = x[:,:,np.newaxis]
    return x.astype(np.float32) / 255 * 2 - 1


def data_generator(batch_size=56):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X,None
                X = []


x_in = Input(shape=(img_dim_h, img_dim_w, 1))
x = x_in
x = Conv2D(32, kernel_size=(5,5), strides=(2,2), padding='SAME', activation='relu')(x)
x = Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='SAME', activation='relu')(x)
x = Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='SAME',activation='relu')(x)
x = Conv2D(256, kernel_size=(3,3), strides=(5,5), padding='SAME',activation='relu')(x)
z_mean = Conv2D(z_dim, kernel_size=(1, 1), strides=(1, 1))(x)
z_mean = GlobalAveragePooling2D()(z_mean)
z_log_var = Conv2D(z_dim, kernel_size=(1, 1), strides=(1, 1))(x)
z_log_var = GlobalAveragePooling2D()(z_log_var)
#x = GlobalAveragePooling2D()(x)

encoder = Model(x_in, outputs=[z_mean,z_log_var])
encoder.summary()
map_size = K.int_shape(encoder.layers[-3].output)[1:-1]

# Decoder pipline
#z_in = Input(shape=K.int_shape(x)[1:])
z_in = Input((z_dim,))
z = z_in
z = Dense(np.prod(map_size)*z_dim)(z)
z = Reshape(map_size + (z_dim,))(z)

z = Conv2D(256, kernel_size=(1,1), strides=(1,1), padding='SAME', activation='relu')(z)
z = Conv2DTranspose(256, kernel_size=(3,3), strides=(2,2), padding='SAME', activation='relu')(z)
z = Conv2DTranspose(128, kernel_size=(5,5), strides=(5,5), padding='SAME', activation='relu')(z)
z = Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2), padding='SAME', activation='relu')(z)
z = Conv2DTranspose(32, kernel_size=(3,3), strides=(2,2), padding='SAME', activation='relu')(z)
z = ReflectionPadding2D((1,1))(z)
z = Conv2D(1, kernel_size=(3,3), strides=(1,1))(z)
z = Activation('tanh')(z)

decoder = Model(z_in, z)
decoder.summary()


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

#z_mean = Dense(z_dim)(x)
#z_log_var = Dense(z_dim)(x)

z = Lambda(sampling, output_shape=(z_dim,))([z_mean, z_log_var])

x_recon = decoder(z)
x_out = Subtract()([x_in, x_recon])

recon_loss = 0.5 * K.sum(K.mean(x_out**2, 0)) + 0.5 * np.log(2*np.pi) * np.prod(K.int_shape(x_out)[1:])
#z_loss = 0.5 * K.sum(K.mean(z**2, 0)) - 0.5 * K.sum(K.mean(u**2, 0))
#recon_loss = 0.5 * K.sum(K.mean(x_out**2, 0)) 
#recon_loss = K.sum(K.binary_crossentropy(x_in,x_recon),axis=[1,2,3])
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = recon_loss + kl_loss

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
    figure = (figure+1)* 255/2 
    imageio.imwrite(path, figure)


class Evaluate(Callback):
    def __init__(self):
        import os
        self.lowest = 1e10
        self.losses = []
        if not os.path.exists('samples_finetune'):
            os.mkdir('samples_finetune')
    def on_epoch_end(self, epoch, logs=None):
        path = 'samples_finetune/test_%s.png' % epoch
        sample(path)
        self.losses.append((epoch, logs['loss']))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('finetune/best_encoder_'+str(epoch)+'.weights')
            decoder.save_weights('finetune/best_decoder'+str(epoch)+'.weights')

vae.save('VAE.h5')
json_string = vae.to_json()
open('model_architecture.json','w').write(json_string)
evaluator = Evaluate()

if finetune:
    encoder.load_weights('model/best_encoder.weights')
    decoder.load_weights('model/best_decoder.weights')

vae.fit_generator(data_generator(),
                  epochs=1000,
                  steps_per_epoch=100,
                  callbacks=[evaluator])

