import numpy as np
from keras.datasets import mnist
#
# import imp
# from path import Path
# cfg_dir = Path("/home/hope-yao/Documents/VAE_SAL/3DVAE_config.py")
# config_module = imp.load_source("3DVAE_config", cfg_dir)
#
# model = config_module.get_model(interp=False)
#
# vae = model['vae']
# vae.summary()
# cfg = config_module.cfg
#
# [x_train,y_train,x_test,y_test] = config_module.get_data(db='mnist')
#
# vae.fit(x_train, x_train,
#         shuffle=True,
#         nb_epoch=cfg['max_epochs'],
#         batch_size=cfg['batch_size'],
#         validation_data=(x_test, x_test))


from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Lambda
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.models import Model
from keras import backend as K

def g2b(g_V):
    '''grey value to binary'''
    return K.round(g_V)


if K.image_dim_ordering() == 'tf':
    original_img_size = (28, 28, 1)
else:
    original_img_size = (1, 28, 28)

g_V1 = Input(shape=original_img_size)
g_V2 = Lambda(g2b, output_shape=g_V1._keras_shape[1:3])(g_V1)

vae = Model(g_V1, g_V2)

vae.compile(optimizer='rmsprop', loss='binary_crossentropy')
vae.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
# x_train[x_train < 0.5] = 0
# x_train[x_train >= 0.5] = 1

# x_test = x_test.astype('float32') / 255.
# x_test = x_test.reshape((x_test.shape[0],) +  original_img_size)
# x_test[x_test < 0.5] = 0
# x_test[x_test >= 0.5] = 1

n = 10
data = vae.predict(x_train[0:n])
x_train1 = x_train[0:n]
x_train1[x_train1 < 0.5] = 0
x_train1[x_train1 >= 0.5] = 1
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(data[i].reshape(28, 28), cmap='Greys', interpolation='nearest')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i + n + 1)

    # display reconstruction
    plt.imshow(x_train1[i].reshape(28, 28), cmap='Greys', interpolation='nearest')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show(block=True)

# vae.fit(x_train, x_train,
#                  shuffle=True,
#                  nb_epoch=1,
#                  batch_size=8,
#                  validation_data=(x_test, x_test))
