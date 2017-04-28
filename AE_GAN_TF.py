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


# problem:
# x_rec ranging from 0.1 to about 0.5, not sure why

import tensorflow as tf
from tqdm import tqdm
import os
import dateutil.tz
import datetime
import math
import numpy as np
from PIL import Image
import json
import sys
# import prettytensor as pt

slim = tf.contrib.slim
TINY = 1e-8

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

def save_config(config, model_dir):
    """code from BEGAN"""
    param_path = os.path.join(model_dir, "params.txt")
    json.dump(config, open(param_path , 'w'))


class VAE(object):
    def __init__(self, cfg):
        self.log_vars = []

        # Network Parameter Setting
        self.variational = cfg['vae']
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
        self.vae_lr = cfg['vae_lr']
        self.snapshot_interval = cfg['snapshot_interval']

        if self.variational:
            self.network_type = 'VAE'
        else:
            self.network_type = 'AE'
        self.logdir,self.modeldir = self.creat_dir(self.network_type)
        save_config(cfg, self.modeldir)

        # define VAE network
        self.x_input = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.n_ch_in])
        self.y_input = tf.placeholder(tf.float32, [self.batch_size, self.n_attributes])
        self.x_rec = self.def_vae(self.x_input)

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

    def activation(self, cfg):
        '''define activation function of the convolution'''
        if cfg['act_func'] == 'ELU':
            alpha = 1.0,  # ELU
            return tf.nn.elu
        elif cfg['act_func'] == 'ReLu':
            alpha = 0.3,  # LeakyReLu
            return tf.nn.relu
        else:
            raise NameError('Not supported activation function type')

    def sampling(self, z_mean, z_log_var ):
        epsilon = tf.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.)
        return z_mean + tf.exp(z_log_var / 2) * epsilon

    def compute_vae_loss(self, x_input, x_rec, variational):
        xent_loss = tf.reduce_mean(-x_input * tf.log(x_rec+ TINY) - (1 - x_input) * tf.log( 1 - x_rec+ TINY))
        rec_er = tf.reduce_mean(tf.squared_difference(x_input, x_rec))
        self.log_vars.append(("xent_loss", tf.reduce_mean(xent_loss)))
        self.log_vars.append(("vae_rec_er", tf.reduce_mean(rec_er)))
        if variational:
            kl_loss = - 0.5 * tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=-1)
            self.log_vars.append(("mean_variance", tf.reduce_mean(self.z_log_var)))
            self.log_vars.append(("kl_loss", tf.reduce_mean(kl_loss)))
        else:
            kl_loss = 0
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
        # initializer = tf.truncated_normal_initializer(stddev=0.01)
        # regularizer = slim.l2_regularizer(0.0005)
        output = slim.conv2d_transpose(output, self.n_ch_in, (3, 3), activation_fn=tf.sigmoid, scope='this')#, padding='same'
        return output

    def basic_enc_conv_module(self, input, n_ch):
        '''convolution module for encoder'''
        initializer = tf.truncated_normal_initializer(stddev=0.01)
        regularizer = slim.l2_regularizer(0.0005)
        output = slim.conv2d(input, n_ch, 3, 1, activation_fn=self.act_func)
        output = slim.conv2d(output, n_ch, 3, 2, activation_fn=self.act_func) #scope='conv3_1'
        return output

    def basic_dec_conv_module(self, input, n_ch):
        '''convolution module for decoder'''
        initializer = tf.truncated_normal_initializer(stddev=0.01)
        regularizer = slim.l2_regularizer(0.0005)
        output = slim.conv2d_transpose(input, n_ch, 3, 1, activation_fn=self.act_func)
        output = slim.conv2d_transpose(output, n_ch, 3, 2, activation_fn=self.act_func) #scope='conv3_1'
        return output

    def def_vae_test(self,x_input):
        n_blocks = 1 # there are n_blocks convolution and pooling structure

        # Encoder
        with tf.variable_scope("encoder", reuse=True):
            output = self.encoder(x_input, n_blocks)
        # Decoder
        with tf.variable_scope("decoder", reuse=True):
            x_rec = self.decoder(output, n_blocks)

        return x_rec

    def def_vae(self,x_input):
        '''build up model'''
        n_blocks = 1 # there are n_blocks convolution and pooling structure

        # Encoder
        with tf.variable_scope("encoder", reuse=True):
            output = self.encoder(x_input, n_blocks)

        # FC layers
        with tf.variable_scope("FC", reuse=True):
            h1 = slim.flatten(output)
            iterm_size = self.img_size/2**n_blocks
            iterm_ch = self.n_filter*2**(n_blocks-1)
            if self.network_type == 'AE':
                h2 = slim.fully_connected(h1, self.latent_dim,activation_fn=tf.nn.relu)
            else:
                h2 = slim.fully_connected(h1, 2*self.latent_dim,activation_fn=tf.sigmoid) # doubled for both mean and variance
                self.z_mean, self.z_log_var = tf.split(h2, num_or_size_splits=2, axis=1)
                h2 =  self.sampling(self.z_mean, self.z_log_var)
            h3 = slim.fully_connected(h2, iterm_size*iterm_size*iterm_ch, activation_fn=tf.nn.relu)
            h = tf.reshape(h3,[self.batch_size, iterm_size , iterm_size , iterm_ch])

        # Decoder
        with tf.variable_scope("decoder", reuse=True):
            x_rec = self.decoder(h, n_blocks)

        return x_rec

    def train_vae(self, **kwargs):
        '''model training'''

        epochs = self.epochs
        for name, value in kwargs.items():
            if name=='epochs':
                epochs = value # overwrite in case of pretraining

        with tf.Session() as sess:

            if self.optimizer == 'adam':
                vae_optimizer = tf.train.AdamOptimizer()
            elif self.optimizer == 'adadelta':
                vae_optimizer = tf.train.AdadeltaOptimizer()
            else:
                raise Exception("[!] Caution! {} opimizer is not implemented in VAE training".format(self.optimizer))
            self.vae_loss = self.compute_vae_loss(self.x_input, self.x_rec, self.variational)
            vae_trainer = vae_optimizer.minimize(self.vae_loss)#, var_list=self.vae_var

            init = tf.global_variables_initializer()
            sess.run(init)
            log_vars = [x for _, x in self.log_vars]
            log_keys = [x for x, _ in self.log_vars]
            for k, v in self.log_vars:
                tf.summary.scalar(k, v)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
            saver = tf.train.Saver()

            counter = 0
            for epoch in tqdm(range(epochs)):
                it_per_ep = len(self.X_train) / self.batch_size
                all_log_vals = []
                for i in range(it_per_ep):  # 1875 * 32 = 60000 -> # of training samples
                    counter += 1
                    x_train = self.X_train[i * self.batch_size:(i + 1) * self.batch_size]
                    y_train = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]
                    feed_dict = {self.x_input: x_train, self.y_input: y_train}
                    log_vals = sess.run([vae_trainer] + log_vars, feed_dict)[1:]
                    all_log_vals.append(log_vals)

                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % ('experiment', str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.modeldir, snapshot_name))
                        print("Model saved in file: %s" % fn)

                x_train = self.X_train[i * self.batch_size:(i + 1) * self.batch_size]
                y_train = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]
                feed_dict = {self.x_input: x_train, self.y_input: y_train}
                summary_str = sess.run(summary_op, feed_dict)
                summary_writer.add_summary(summary_str, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                print("Epoch %d | " % (epoch) + log_line)
                sys.stdout.flush()
                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")

                # save reconstructed images
                x_rec = self.def_vae(x_train)
                x_rec_rec = self.def_vae(x_rec)
                all_G_z = np.concatenate([255 * x_train[0:8], 255 * x_rec.eval()[0:8], 255 * x_rec_rec.eval()[0:8]])
                save_image(all_G_z, '{}/epoch_{}_{}.png'.format(self.logdir, epoch, log_line))


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

class GAN(VAE):
    def __init__(self,cfg):
        super(GAN,self).__init__(cfg)

        # Working and saving dir
        self.network_type = 'GAN'
        self.logdir,self.modeldir = self.creat_dir(self.network_type)
        save_config(cfg, self.modeldir)

        # Network parameters for GAN
        self.d_lr = cfg['d_lr']
        self.g_lr = cfg['g_lr']
        self.d_optimizer = tf.train.AdamOptimizer(self.d_lr)
        self.g_optimizer = tf.train.AdamOptimizer(self.g_lr)

        self.k_t = 1  # will be modified into PID controller in the future

        if cfg['pre_train']:
            self.train_vae(epochs=1)

    def get_d_loss(self,x_input):
        # real_rec_loss = objectives.binary_crossentropy(x_input, x_gen)
        # fake_rec_loss = objectives.binary_crossentropy(x_gen, x_gen_gen)
        x_rec = self.def_vae(x_input)
        x_rec_rec = self.def_vae(x_rec)
        d_loss_real = tf.reduce_mean(tf.abs(x_input - x_rec))
        d_loss_fake = tf.reduce_mean(tf.abs(x_rec - x_rec_rec))
        d_loss = d_loss_real - self.k_t* d_loss_fake
        return d_loss

    def get_g_loss(self,x_input):
        x_rec = self.def_vae(x_input)
        x_rec_rec = self.def_vae(x_rec)
        d_loss_fake = tf.reduce_mean(tf.abs(x_rec - x_rec_rec))
        g_loss = d_loss_fake
        return g_loss
    #     # x = self.norm_img(x_train)
    #     # self.z = tf.random_uniform(
    #     #     (tf.shape(x)[0], self.z_num), minval=-1.0, maxval=1.0)
    #     # self.decoder(z)
    #     # self.k_t = tf.Variable(0., trainable=False, name='k_t')
    def train_gan(self, **kwargs):
        '''model training'''
        epochs = self.epochs

        with tf.Session() as sess:

            self.d_loss = self.get_d_loss(self.x_input)
            self.g_loss = self.get_g_loss(self.x_input)
            self.log_vars.append(("d_loss", tf.reduce_mean(self.d_loss)))
            self.log_vars.append(("g_loss", tf.reduce_mean(self.g_loss)))
            d_trainer = self.d_optimizer.minimize(self.d_loss)  # , var_list=self.vae_var
            g_trainer = self.d_optimizer.minimize(self.g_loss)  # , var_list=self.vae_var

            init = tf.global_variables_initializer()
            sess.run(init)
            log_vars = [x for _, x in self.log_vars]
            log_keys = [x for x, _ in self.log_vars]
            for k, v in self.log_vars:
                tf.summary.scalar(k, v)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
            saver = tf.train.Saver()

            counter = 0
            for epoch in tqdm(range(epochs)):
                it_per_ep = len(self.X_train) / self.batch_size
                all_log_vals = []
                for i in range(it_per_ep):  # 1875 * 32 = 60000 -> # of training samples
                    counter += 1
                    x_train = self.X_train[i * self.batch_size:(i + 1) * self.batch_size]
                    y_train = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]
                    feed_dict = {self.x_input: x_train, self.y_input: y_train}
                    log_vals = sess.run([g_trainer] + log_vars, feed_dict)[1:]
                    log_vals = sess.run([d_trainer] + log_vars, feed_dict)[1:]
                    all_log_vals.append(log_vals)

                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % ('experiment', str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.modeldir, snapshot_name))
                        print("Model saved in file: %s" % fn)

                x_train = self.X_train[i * self.batch_size:(i + 1) * self.batch_size]
                y_train = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]
                feed_dict = {self.x_input: x_train, self.y_input: y_train}
                summary_str = sess.run(summary_op, feed_dict)
                summary_writer.add_summary(summary_str, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                print("Epoch %d | " % (epoch) + log_line)
                sys.stdout.flush()
                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")

                # save reconstructed images
                x_rec = self.def_vae(x_train)
                x_rec_rec = self.def_vae(x_rec)
                all_G_z = np.concatenate([255 * x_train[0:8], 255 * x_rec.eval()[0:8], 255 * x_rec_rec.eval()[0:8]])
                save_image(all_G_z, '{}/epoch_{}_{}.png'.format(self.logdir, epoch, log_line))

                    # with tf.control_dependencies([d_optim, g_optim]):
            #     self.k_update = tf.assign(
            #         self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

if __name__ == "__main__":

    cfg = {'batch_size': 64,
           'act_func': 'ReLu', #ELU, ReLu
           'input_dim': (64, 64),
           'n_channels': 3,
           'n_attributes': 40,
           'n_filters': 16,
           'filter_size': (3,3),
           # 'n_classes': 10,
           'max_epochs': 1000,
           'latent_dim': 100,
           'vae_optimizer': 'adadelta',
           'vae_lr': 1e-1,
           'g_lr': 5e-3,
           'd_lr': 1e-4,
           'g_optimizer': 'adam',
           'd_optimizer': 'sgd',
           # 'learning_rate': lr_schedule,
           'vae': False,
           'datadir': '/home/hope-yao/Documents/Data',
           'pre_train': True,
           'snapshot_interval': 5000,
           }

    vae = VAE(cfg)
    vae.train_vae()

    # gan = GAN(cfg)
    # gan.train_gan()




