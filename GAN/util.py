
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

