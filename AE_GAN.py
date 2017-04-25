# TO DO:
# Add tensorflow image monitoring

import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D, Lambda
from keras.models import Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras import objectives
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint

class VAE(object):
    def __init__(self,cfg):
        self.variational = cfg['vae']

        # Network Parameter Setting
        self.act_func = self.activation(cfg)
        self.latent_dim = cfg['latent_dim']
        self.n_filter = cfg['n_filters']
        self.filter_size = cfg['filter_size']
        self.img_size = cfg['input_dim'][0]
        self.n_ch_in = cfg['n_channels']
        self.x_input = Input(shape=(self.img_size, self.img_size, self.n_ch_in))
        self.encoder_module = self.basic_enc_conv_module
        self.decoder_module = self.basic_dec_conv_module

        # Optimizer Parameter Setting
        self.batch_size = cfg['batch_size']
        self.epochs = cfg['max_epochs']
        self.optimizer = cfg['optimizer']
        self.get_data = self.CelebA()

        self.vae_model = self.def_model()

    def CelebA(self):
        '''load human face dataset'''
        import h5py
        from random import sample
        import numpy as np
        f = h5py.File("/home/hope-yao/Documents/Data/celeba.hdf5", "r")
        data_key = f.keys()[0]
        data = np.asarray(f[data_key],dtype='float32') / 255.
        data = data.transpose((0,2,3,1))
        label_key = f.keys()[1]
        label = np.asarray(f[label_key])

        split = 0.1
        l = len(data)  # length of data
        n1 = int(split * l)  # split for testing
        indices = sample(range(l), n1)

        x_test = data[indices]
        y_test = label[indices]
        x_train = np.delete(data, indices, 0)
        y_train = np.delete(label, indices, 0)

        return (x_train, y_train), (x_test, y_test)
        # return (x_train[0:10000], y_train[0:10000]), (x_test[0:1000], y_test[0:1000])

    def Inception_module(self, input_img):
        tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)
        tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)

        tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)
        tower_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(tower_2)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
        tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)

        output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

        return output

    def activation(self, cfg):
        '''define activation function of the convolution'''
        if cfg['act_func'] == 'ELU':
            alpha = 1.0,  # ELU
            return ELU(alpha=alpha)
        elif cfg['act_func'] == 'LeakyReLu':
            alpha = 0.3,  # LeakyReLu
            return LeakyReLU(alpha = alpha)
        else:
            raise NameError('Not supported activation function type')

    def basic_enc_conv_module(self, input, n_ch):
        '''convolution module for encoder'''
        output = self.act_func(Conv2D(n_ch, self.filter_size, padding='same')(input))
        output = self.act_func(Conv2D(n_ch, self.filter_size, padding='same')(output))
        output = MaxPooling2D((2, 2), border_mode='same')(output)
        return output

    def basic_dec_conv_module(self, input, n_ch):
        '''convolution module for decoder'''
        output = self.act_func(Conv2D(n_ch, self.filter_size, padding='same')(input))
        output = self.act_func(Conv2D(n_ch, self.filter_size, padding='same')(output))
        output = UpSampling2D((2, 2))(output)
        return output

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def ae_loss(self, x_input, x_rec):
        xent_loss = objectives.binary_crossentropy(x_input, x_rec)
        return xent_loss

    def vae_loss(self, x_input, x_rec):
        xent_loss = objectives.binary_crossentropy(x_input, x_rec)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss

    def def_vae(self):
        '''build up model'''
        n_blocks = 3 # there are n_blocks convolution and pooling structure

        # Encoder
        output = self.x_input
        for block_i in range(n_blocks):
            n_ch = self.n_filter * 2 ** block_i
            output = self.encoder_module(output, n_ch)

        # FC layers
        h1 = Flatten()(output)
        iterm_size = self.img_size/2**n_blocks
        iterm_ch = self.n_filter*2**(n_blocks-1)
        if ~self.variational:
            h2 = Dense(self.latent_dim, activation='sigmoid')(h1)
        else:
            h2 = Dense(2*self.latent_dim, activation='sigmoid')(h1) # doubled for both mean and variance
            self.z_mean, self.z_log_var = tf.split(h2, num_or_size_splits=2, axis=1)
            h2 = Lambda(lambda x: self.sampling([self.z_mean, self.z_log_var]))
        h3 = Dense(iterm_size*iterm_size*iterm_ch, activation='sigmoid')(h2)
        h = Reshape((iterm_size , iterm_size , iterm_ch))(h3)

        # Decoder
        output = h
        for block_i in range(n_blocks,0,-1):
            n_ch = self.n_filter*2**(block_i-1)
            output = self.decoder_module(output, n_ch)
        self.x_rec = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(output)

        vae_model = Model(self.x_input,self.x_rec)
        vae_model.summary()
        return vae_model

    def train_vae(self):
        '''model training'''
        if ~self.variational:
            self.vae_model.compile(optimizer=self.optimizer, loss=self.ae_loss)
        else:
            self.vae_model.compile(optimizer=self.optimizer, loss=self.vae_loss)
        (x_train, y_train), (x_test, y_test) = self.get_data

        monitor = [TensorBoard(log_dir='./logs'),
                   ModelCheckpoint(filepath='./model/weights.{epoch:02d}-{val_loss:.3f}.hdf5', mode='auto')]
        self.vae_model.fit(x_train, x_train, shuffle=True,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      validation_data=(x_test, x_test),
                      callbacks = monitor)
        return self.vae_model

    def test_vae(self):
        from keras.models import load_model
        vae = load_model('vae_celeba.h5', custom_objects={'vae_loss': self.vae_loss})
        (x_train, y_train), (x_test, y_test) = self.get_data
        x_rec = vae.predict(x_test)
        self.plt_rec(x_test,x_rec)

    def plt_rec(self, x_test, x_rec):
        import matplotlib.pyplot as plt
        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(64, 64, 3))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(2, n, i + n + 1)

            # display reconstruction
            plt.imshow(x_rec[i].reshape(64, 64, 3))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


class GAN(VAE):
    def __init__(self,cfg):
        super(GAN,self).__init__(cfg)

    def nchw_to_nhwc(self, x):
        return tf.transpose(x, [0, 2, 3, 1])

    def norm_img(self, image, data_format=None):
        image = image / 127.5 - 1.
        if data_format:
            image = self.to_nhwc(image, data_format)
        return image

    def to_nhwc(self, image, data_format):
        if data_format == 'NCHW':
            new_image = self.nchw_to_nhwc(image)
        else:
            new_image = image
        return new_image

    def train_gan(self):
        (x_train, y_train), (x_test, y_test) = self.get_data

        x = self.norm_img(x_train)

        self.z = tf.random_uniform(
            (tf.shape(x)[0], self.z_num), minval=-1.0, maxval=1.0)
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        G, self.G_var = GeneratorCNN(
            self.z, self.conv_hidden_num, self.channel,
            self.repeat_num, self.data_format, reuse=False)

        d_out, self.D_z, self.D_var = DiscriminatorCNN(
            tf.concat([G, x], 0), self.channel, self.z_num, self.repeat_num,
            self.conv_hidden_num, self.data_format)
        AE_G, AE_x = tf.split(d_out, 2)

        self.G = denorm_img(G, self.data_format)
        self.AE_G, self.AE_x = denorm_img(AE_G, self.data_format), denorm_img(AE_x, self.data_format)


if __name__ == "__main__":

    cfg = {'batch_size': 64,
           'act_func': 'ELU', #ELU, LeakyReLu
           'input_dim': (64, 64),
           'n_channels': 3,
           'n_filters': 16,
           'filter_size': (3,3),
           # 'n_classes': 10,
           'max_epochs': 30,
           'latent_dim': 100,
           'optimizer': 'adadelta',
           # 'learning_rate': lr_schedule,
           'vae': True,
           }
    gan = GAN(cfg)
    vae = VAE(cfg)
    vae.train_model()
    vae.test_model()



