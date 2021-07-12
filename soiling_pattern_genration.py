#! -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES']=''


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




img_dim_h = 240
img_dim_w = 320
z_dim = 100




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
map_size = K.int_shape(encoder.layers[-3].output)[1:-1]

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


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(z_dim,))([z_mean, z_log_var])

x_recon = decoder(z)
x_out = Subtract()([x_in, x_recon])
vae = Model(x_in, x_out)

encoder.load_weights('finetune/best_encoder_15.weights')
decoder.load_weights('finetune/best_decoder.weights')



def latent_interpolation(encoder,decoder,A,B):
    f_A = A.astype(np.float32) / 255 * 2 - 1
    f_B = B.astype(np.float32) / 255 * 2 - 1
    f_A = f_A[np.newaxis, :]
    A_mean,A_var = encoder.predict(f_A)
    z_a = np.random.randn(100)
    z_b = z_a
    latent_A = A_mean + np.exp(A_var / 2) * z_a
    

    f_B = f_B[np.newaxis, :]
    B_mean,B_var = encoder.predict(f_B)
    latent_B = B_mean + np.exp(B_var / 2) * z_b

    alpha = [0,0.2,0.4,0.6,0.8,1.0]

    concat_img = np.zeros((img_dim_h,img_dim_w*len(alpha)))
    for i in range(len(alpha)):
        z_intermediate = alpha[i]*latent_A+(1-alpha[i])*latent_B
        x_interpolation = decoder.predict(z_intermediate)
        x_interpolation = (x_interpolation+1) * 127.5
        concat_img[0:img_dim_h,i*img_dim_w:(i+1)*img_dim_w] = x_interpolation[0][:,:,0]
        cv2.imwrite(str(i)+'.jpg',x_interpolation[0][:,:,0])
    cv2.imwrite('interpolation.jpg',concat_img)
    print(i)
    

    return None


ImageA_path = '4962_MVR.png'
ImageB_path = '4971_RV.png'

A_pic = cv2.imread(ImageA_path)
B_pic = cv2.imread(ImageB_path)
A_pic = cv2.resize(A_pic,(img_dim_w,img_dim_h))
B_pic = cv2.resize(B_pic,(img_dim_w,img_dim_h))
latent_interpolation(encoder,decoder,A_pic[:,:,1],B_pic[:,:,1])