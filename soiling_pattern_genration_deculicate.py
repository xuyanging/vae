#! -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import os
import cv2
from tensorflow.python.ops import image_ops
os.environ['CUDA_VISIBLE_DEVICES']=''





img_dim_h = 240
img_dim_w = 320
z_dim = 512




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

vae = Model(x_in, x_out)

encoder.load_weights('Ori_model/best_encoder.weights')
decoder.load_weights('Ori_model/best_decoder.weights')
All_model = load_model('Ori_model/best_model.h5',custom_objects={'ScaleShift':ScaleShift})



def latent_interpolation(model,encoder,decoder,A,B):
    f_A = A.astype(np.float32) / 255 
    f_B = B.astype(np.float32) / 255 
    f_A = f_A[np.newaxis, :]
    layer_shift = Model(inputs = model.input, outputs = model.layers[17].output)
    A_shift = layer_shift.predict(f_A)
    layer_log_scale = Model(inputs = model.input, outputs = model.layers[19].output)
    A_log_scale = layer_log_scale.predict(f_A)
    z_a = np.random.randn(512)
    latent_A = A_shift + np.exp(A_log_scale / 2) * z_a
    
    f_B = f_B[np.newaxis, :]
    
    B_shift = layer_shift.predict(f_B)
    B_log_scale = layer_log_scale.predict(f_B)
    z_b = np.random.randn(512)
    latent_B = B_shift + np.exp(B_log_scale / 2) * z_b

    alpha = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    concat_img = np.zeros((img_dim_h,img_dim_w*len(alpha)))
    for i in range(len(alpha)):
        z_intermediate = alpha[i]*latent_A+(1-alpha[i])*latent_B
        x_interpolation = decoder.predict(z_intermediate)
        x_interpolation = x_interpolation * 255
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
latent_interpolation(All_model,encoder,decoder,A_pic,B_pic)