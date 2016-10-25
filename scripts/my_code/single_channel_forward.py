import numpy as np

import lasagne
import lasagne.layers
from lasagne.utils import floatX
from voxnet import isovox

import voxnet
import theano
import theano.tensor as T
from voxnet import npytar

import matplotlib.pyplot as plt

cfg = {'batch_size' : 10, # previous 32
       # 'learning_rate' : lr_schedule, #doesn't matter for forward
       'reg' : 0.001,
       'momentum' : 0.9,
       'dims' : (32, 32, 32),
       'n_channels' : 1,
       'n_classes' : 2,# previous 10
       'batches_per_chunk': 1, # previous 64
       'max_epochs' : 600,  #previous 80
       'max_jitter_ij' : 2,
       'max_jitter_k' : 2,
       'n_rotations' : 1, # previous 12
       'checkpoint_every_nth' : 4000,
       }


# same network structure as Voxnet, only without dropout
dims, n_channels, n_classes = tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes']
shape = (None, n_channels)+dims
l_in = lasagne.layers.InputLayer(shape=shape)
l_conv1 = voxnet.layers.Conv3dMMLayer(
        input_layer = l_in,
        num_filters = 1, # previously 32
        filter_size = [5,5,5],
        border_mode = 'valid',
        strides = [2,2,2],
        # W = voxnet.init.Prelu(),
        W = voxnet.init.Ones(),
        # W = voxnet.init.loadw1(),
        nonlinearity = voxnet.activations.leaky_relu_01,
        name =  'conv1',
        # b = floatX(np.zeros(l_in.shape[1]))
    )
l_conv2 = voxnet.layers.Conv3dMMLayer(
    input_layer = l_conv1,
    num_filters = 1, # previously 32
    filter_size = [3,3,3],
    border_mode = 'valid',
    # W = voxnet.init.Prelu(),
    W=voxnet.init.Ones(),
    # W=voxnet.init.loadw2(),
    nonlinearity = voxnet.activations.leaky_relu_01,
    name = 'conv2',
    # b = floatX(np.zeros(l_conv1.output_shape[1]))
)
l_pool2 = voxnet.layers.MaxPool3dLayer(
    input_layer = l_conv2,
    pool_shape = [2,2,2],
    name = 'pool2',
    )
l_fc0 = lasagne.layers.FlattenLayer(l_pool2, name = 'fc0')
l_fc1 = lasagne.layers.DenseLayer(
    incoming = l_fc0,
    num_units = 1, # previously 128
    # W = lasagne.init.Normal(std=0.01),
    W=voxnet.init.fcwt(),
    name =  'fc1'
    )
l_fc2 = lasagne.layers.DenseLayer(
    incoming = l_fc1,
    num_units = n_classes,
    # W = lasagne.init.Normal(std = 0.01),
    W=voxnet.init.Ones(),
    nonlinearity = None,
    name = 'fc2'
    )

# load data
def data_loader(cfg, fname):
    dims = cfg['dims']
    chunk_size = cfg['n_rotations']
    xc = np.zeros((chunk_size, cfg['n_channels'],)+dims, dtype=np.float32)
    reader = npytar.NpyTarReader(fname)
    yc = []
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.float32)
        yc.append(int(name.split('.')[0])-1)
        if len(yc) == chunk_size:
            yield (xc, np.asarray(yc, dtype=np.float32))
            yc = []
            xc.fill(0)
    assert(len(yc)==0)


# forward pass
layers = [l_in, l_conv1, l_conv2, l_pool2, l_fc0, l_fc1, l_fc2]
# change number here to visualize activations at different layer
layer_idx = 4
l_out = layers[layer_idx]
X = T.TensorType('float32', [False] * 5)('X')
act = lasagne.layers.get_output(l_out, X, deterministic=True)
tt = theano.function([X], act)

loader = (data_loader(cfg, '../../more_data_real_100_100_1_0/shapenet10_train.tar'))
rrtt0 = [] # for pot
rrtt1 = [] # for cup
for i,(x_shared, y_shared) in enumerate(loader):
    rr = tt(x_shared)
    if y_shared[0] == 0:
        rrtt0.append( rr )
    else:
        rrtt1.append( rr )

    if l_out.name in {'fc0','fc1','fc2'}:
        print(y_shared[0], rr[0])
    else:
        size = 32
        w = rr[0, 0]
        # centerize the plot
        fz = len(w)
        xd = np.zeros((size,size,size))
        pad = (size-fz)/2
        xd[pad:pad+fz,pad:pad+fz,pad:pad+fz] = w
        # only visualize the largest value
        t = 0
        xd[xd<t]=0
        # store as png
        iv = isovox.IsoVox()
        img = iv.render(xd, as_html=True, name='../act/inst'+str(i))
        print(y_shared[0], rr[0])
    if l_out.name == 'fc2':
        if y_shared[0] == 0:
            plt.plot(rr[0, 0], rr[0, 1], 'b.')
        else:
            plt.plot(rr[0, 0], rr[0, 1], 'ro')
    if l_out.name == 'fc1':
        if y_shared[0] == 0:
            plt.loglog(rr[0, 0], rr[0, 0], 'b.')
        else:
            plt.loglog(rr[0, 0], rr[0, 0], 'ro')

loader1 = (data_loader(cfg, '../../more_data_real_100_100_1_0/shapenet10_test.tar'))
for i,(x_shared, y_shared) in enumerate(loader1):
    rr = tt(x_shared)
    if l_out.name == 'fc1':
        if y_shared[0] == 0:
            plt.loglog(rr[0, 0], rr[0, 0], 'bv')
        else:
            plt.loglog(rr[0, 0], rr[0, 0], 'r^')

if l_out.name == 'fc0':
    fig, axes = plt.subplots(nrows=2, ncols=1)

    loader1 = (data_loader(cfg, '../../more_data_real_10_10_1_0/shapenet10_test.tar'))
    for i, (x_shared, y_shared) in enumerate(loader1):
        rr = tt(x_shared)
        if y_shared[0] == 0:
            rrtt0.append(rr)
        else:
            rrtt1.append(rr)

    ax1 = axes[0]
    im = ax1.imshow(np.squeeze(np.asarray(rrtt0)))
    ax1.set_title("training POT activation before going to fc1")

    ax2 = axes[1]
    im = ax2.imshow(np.squeeze(np.asarray(rrtt1)))
    ax2.set_title(" CUP activation before going to fc1")

    fig.colorbar(im, ax=axes.ravel().tolist())
    # fig.savefig('test.png')
    plt.show()
if l_out.name in {'fc1','fc2'}:
    plt.show()
