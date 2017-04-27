
class VAE(object):
    def __init__(self, cfg):
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

    def train_vae(self, **kwargs):
        '''model training'''
        self.vae_model.summary()

        if ~self.variational:
            self.vae_model.compile(optimizer=self.optimizer, loss=self.ae_loss)
        else:
            self.vae_model.compile(optimizer=self.optimizer, loss=self.vae_loss)

        monitor = [TensorBoard(log_dir='./logs'),
                   ModelCheckpoint(filepath='./model/VAE.{epoch:02d}-{val_loss:.3f}.hdf5', mode='auto')]

        epochs = self.epochs
        for name, value in kwargs.items():
            if name=='epochs':
                epochs = value # overwrite in case of pretraining

        self.vae_model.fit(self.X_train, self.X_train, shuffle=True,
                      epochs=epochs,
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
