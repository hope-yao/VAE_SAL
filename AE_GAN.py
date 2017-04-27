# TO DO:
# Add tensorflow image monitoring
# load data in batch to reduce gpu memory consumption
# added Enc(Dec(x)), but need Dec(z)

# add batch norm in vae
# consider dropout to improve GAN
# consider flip labels to relief gradient vanishing
# consider use PID to balance D and G loss
# try inception in vae
# Maybe pre-train GAN using VAE?
# crossentropy loss v.s. l1 norm in VAE loss?

import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D, Lambda
from keras.models import Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras import objectives
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras.engine.topology import Layer
from tqdm import tqdm
import os
import dateutil.tz
import datetime
import math
import numpy as np
from PIL import Image
import json

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

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
        self.n_attributes = cfg['n_attributes']
        self.datadir = cfg['datadir']
        self.encoder_module = self.basic_enc_conv_module
        self.decoder_module = self.basic_dec_conv_module

        # Optimizer Parameter Setting
        self.batch_size = cfg['batch_size']
        self.epochs = cfg['max_epochs']
        self.optimizer = cfg['vae_optimizer']

        # define VAE network
        self.x_input = Input(shape=(self.img_size, self.img_size, self.n_ch_in))
        self.x_rec = self.def_vae(self.x_input)
        self.vae_model = Model(inputs=self.x_input, outputs=self.x_rec)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.CelebA(self.datadir)

    def CelebA(self, datadir):
        '''load human face dataset'''
        import h5py
        from random import sample
        import numpy as np
        f = h5py.File(datadir+"/celeba.hdf5", "r")
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

        # return (x_train, y_train), (x_test, y_test)
        return (x_train[0:10000], y_train[0:10000]), (x_test[0:1000], y_test[0:1000])

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

    def encoder(self, input, n_blocks):
        output = input
        for block_i in range(n_blocks):
            n_ch = self.n_filter * 2 ** block_i
            output = self.encoder_module(output, n_ch)
        return output

    def decoder(self, input, n_blocks):
        output = input
        for block_i in range(n_blocks,0,-1):
            n_ch = self.n_filter*2**(block_i-1)
            output = self.decoder_module(output, n_ch)
        output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(output)
        return output

    def def_vae(self,x_input):
        '''build up model'''
        n_blocks = 3 # there are n_blocks convolution and pooling structure

        # Encoder
        output = self.encoder(x_input, n_blocks)

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
        x_rec = self.decoder(h, n_blocks)

        return x_rec

    def train_vae(self):
        '''model training'''
        self.vae_model.summary()

        if ~self.variational:
            self.vae_model.compile(optimizer=self.optimizer, loss=self.ae_loss)
        else:
            self.vae_model.compile(optimizer=self.optimizer, loss=self.vae_loss)

        monitor = [TensorBoard(log_dir='./logs'),
                   ModelCheckpoint(filepath='./model/VAE.{epoch:02d}-{val_loss:.3f}.hdf5', mode='auto')]
        self.vae_model.fit(self.X_train, self.X_train, shuffle=True,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      validation_data=(self.X_test, self.X_test),
                      callbacks = monitor)
        # self.vae_model.fit_generator(self.myGenerator(),steps_per_epoch=1000)
        return self.vae_model

    def creat_dir(self,network_type):
        """code from on InfoGAN"""
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        root_log_dir = "logs/" + network_type
        exp_name = network_type + "_%s" % timestamp
        log_dir = os.path.join(root_log_dir, exp_name)

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        root_model_dir = "models/" + network_type
        exp_name = network_type + "_%s" % timestamp
        model_dir = os.path.join(root_model_dir, exp_name)

        for path in [log_dir, model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
        return log_dir, model_dir


def test_vae(vae, fn):
    vae.vae_model.load_weights(fn)
    (_, _), (x_test, y_test) = vae.get_data
    x_rec = vae.vae_model.predict(x_test)
    save_image( x_test, x_rec)

def plt_rec(x_test, x_rec):
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

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):
    """code from on BEGAN"""
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)

def concat(args):
    '''keras tensor concatenation'''
    x, y = args
    return tf.concat([x,y],axis=3)


class GAN(VAE):
    def __init__(self,cfg):
        super(GAN,self).__init__(cfg)

        # Working and saving dir
        self.network_type = 'GAN'
        self.logdir,self.modeldir = self.creat_dir(self.network_type)

        # Network parameters for GAN
        self.x_input = Input(shape=(self.img_size, self.img_size, self.n_ch_in))
        self.y_input = Input(shape=(self.n_attributes,))
        self.d_optimizer = SGD(lr=0.001, momentum=0.999, decay=0.,nesterov=False)
        self.g_optimizer = Adam()

        self.x_rec = self.def_vae(self.x_input)
        # x_rec_rec = self.def_vae(x_rec) # This will creat another set of new layers
        self.x_rec_rec = Lambda(self.def_vae, output_shape=(self.img_size,self.img_size,self.n_ch_in))(self.x_rec)

    def d_loss(self, args):
        x_input, x_rec, x_rec_rec = args

        def get_d_loss(self,x_input):
            # real_rec_loss = objectives.binary_crossentropy(x_input, x_gen)
            # fake_rec_loss = objectives.binary_crossentropy(x_gen, x_gen_gen)
            d_loss_real = tf.reduce_mean(tf.abs(x_input - x_rec))
            d_loss_fake = tf.reduce_mean(tf.abs(x_rec - x_rec_rec))
            k_t = 1 # will be modified into PID controller in the future
            d_loss = d_loss_real - k_t * d_loss_fake
            return d_loss

        return get_d_loss

    def g_loss(self, args):
        x_input, x_rec, x_rec_rec = args

        def get_g_loss(self,x_input):
            d_loss_fake = tf.reduce_mean(tf.abs(x_rec - x_rec_rec))
            g_loss = d_loss_fake
            return g_loss

        return get_g_loss

    # def def_gan(self,x_input):
    #     # x = self.norm_img(x_train)
    #     # self.z = tf.random_uniform(
    #     #     (tf.shape(x)[0], self.z_num), minval=-1.0, maxval=1.0)
    #     # self.decoder(z)
    #     # self.k_t = tf.Variable(0., trainable=False, name='k_t')
    #
    #     x_rec = self.def_vae(x_input)
    #     # x_rec_rec = self.def_vae(x_rec) # This will creat another set of new layers
    #     x_rec_rec = Lambda(self.def_vae, output_shape=(self.img_size,self.img_size,self.n_ch_in))(x_rec)
    #
    #     return x_rec, x_rec_rec

    def train_gan(self):

        self.g_net = Model(inputs=self.x_input, outputs=self.x_rec)
        self.g_net.compile(loss=self.g_loss([self.x_input, self.x_rec, self.x_rec_rec]), optimizer=self.g_optimizer)
        self.g_net.summary()

        self.d_net = Model(inputs=self.x_input, outputs=self.x_rec_rec)
        self.d_net.compile(loss=self.d_loss([self.x_input, self.x_rec, self.x_rec_rec]), optimizer=self.d_optimizer)
        self.d_net.summary()

        for e in tqdm(range(self.epochs)):
            it_per_ep = len(self.X_train) / self.batch_size
            for i in range(it_per_ep):  # 1875 * 32 = 60000 -> # of training samples
                x_train = self.X_train[i * self.batch_size:(i + 1) * self.batch_size]
                self.d_net.train_on_batch(x_train, x_train)
                self.g_net.train_on_batch(x_train, x_train)

            x_rec = self.vae_model.predict(x_train)
            x_rec_rec = self.vae_model.predict(x_rec)
            d_loss_real, d_loss_fake = np.mean(np.abs(x_train - x_rec)), np.mean(np.abs(x_rec_rec - x_rec))
            d_loss = d_loss_real - d_loss_fake
            g_loss = d_loss_fake
            print(d_loss_real, d_loss_fake, d_loss, g_loss)

            x_test = self.X_test
            x_rec = self.vae_model.predict(x_test)
            x_rec_rec = self.vae_model.predict(x_rec)
            d_loss_real, d_loss_fake = np.mean(np.abs(x_test - x_rec)), np.mean(np.abs(x_rec_rec - x_rec))
            d_loss = d_loss_real - d_loss_fake
            g_loss = d_loss_fake
            print(d_loss_real, d_loss_fake, d_loss, g_loss)

            all_G_z = np.concatenate([255 * x_train[0:8], 255 * x_rec[0:8], 255 * x_rec_rec[0:8]])
            save_image(all_G_z, '{}/epoch_{}.png'.format(self.logdir, e))
            self.g_net.save('{}/epoch_{}.h5'.format(self.modeldir, e))

def save_config(config):
    """code from BEGAN"""
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def make_trainable(net, val):
    '''Freeze weights in the discriminator for stacked training'''
    # https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN_v2.ipynb
    net.trainable = val
    for l in net.layers:
        l.trainable = val

if __name__ == "__main__":

    cfg = {'batch_size': 64,
           'act_func': 'ELU', #ELU, LeakyReLu
           'input_dim': (64, 64),
           'n_channels': 3,
           'n_attributes': 40,
           'n_filters': 16,
           'filter_size': (3,3),
           # 'n_classes': 10,
           'max_epochs': 100,
           'latent_dim': 100,
           'vae_optimizer': 'adadelta',
           'g_optimizer': 'adam',
           'd_optimizer': 'sgd',
           # 'learning_rate': lr_schedule,
           'vae': False,
           'datadir': '/home/hope-yao/Documents/Data',
           }

    # vae = VAE(cfg)
    # vae.train_vae()
    # test_vae(vae,'./result/AE.99-0.501.hdf5')

    gan = GAN(cfg)
    gan.train_gan()




