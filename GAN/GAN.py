
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

        if cfg['pre_train']:
            self.train_vae(epochs=5)

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

            # output training result
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
           'vae': True,
           'datadir': '/home/hope-yao/Documents/Data',
           'pre_train': True,
           }

    # vae = VAE(cfg)
    # vae.train_vae()
    # test_vae(vae,'./result/AE.99-0.501.hdf5')

    gan = GAN(cfg)
    gan.train_gan()


