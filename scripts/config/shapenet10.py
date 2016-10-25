
import numpy as np

import lasagne
import lasagne.layers

import voxnet

lr_schedule = { 0: 0.001,
                60000: 0.0001,
                400000: 0.00005,
                600000: 0.00001,
                }
# cfg = {'batch_size' : 4, # previous 32
#        'learning_rate' : lr_schedule,
#        'reg' : 0.001,
#        'momentum' : 0.9,
#        'dims' : (32, 32, 32),
#        'n_channels' : 1,
#        'n_classes' : 2,# previous 10
#        'batches_per_chunk': 4, # previous 64
#        'max_epochs' : 8,  #previous 80
#        'max_jitter_ij' : 2,
#        'max_jitter_k' : 2,
#        'n_rotations' : 1, # previous 12
#        'checkpoint_every_nth' : 4000,
#        }

cfg = {'batch_size' : 32, # previous 32
       'learning_rate' : lr_schedule,
       'reg' : 0.001,
       'momentum' : 0.9,
       'dims' : (32, 32, 32),
       'n_channels' : 1,
       'n_classes' : 10,# previous 10
       'batches_per_chunk': 64, # previous 64
       'max_epochs' : 80,  #previous 80
       'max_jitter_ij' : 2,
       'max_jitter_k' : 2,
       'n_rotations' : 12, # previous 12
       'checkpoint_every_nth' : 4000,
       }


def get_model():
    dims, n_channels, n_classes = tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes']
    shape = (None, n_channels)+dims

    l_in = lasagne.layers.InputLayer(shape=shape)
    l_conv1 = voxnet.layers.Conv3dMMLayer(
            input_layer = l_in,
            num_filters = 32, # previously 32
            filter_size = [5,5,5],
            border_mode = 'valid',
            strides = [2,2,2],
            W = voxnet.init.Prelu(),
            nonlinearity = voxnet.activations.leaky_relu_01,
            name =  'conv1'
        )

    l_drop1 = lasagne.layers.DropoutLayer(
        incoming = l_conv1,
        p = 0.2,
        name = 'drop1'
        )
    l_conv2 = voxnet.layers.Conv3dMMLayer(
        input_layer = l_drop1,
        num_filters = 32, # previously 32
        filter_size = [3,3,3],
        border_mode = 'valid',
        W = voxnet.init.Prelu(),
        nonlinearity = voxnet.activations.leaky_relu_01,
        name = 'conv2'
        )

    # Hope added. out put filter for visualization
    visualize_filter = 0
    if visualize_filter:
        W1 = np.array(l_conv1.W.eval())
        np.save('W1.npy',W1)
        W2 = np.array(l_conv2.W.eval())
        np.save('W2.npy', W2)
        import sys
        sys.exit("filters written done")

    l_pool2 = voxnet.layers.MaxPool3dLayer(
        input_layer = l_conv2,
        pool_shape = [2,2,2],
        name = 'pool2',
        )
    l_drop2 = lasagne.layers.DropoutLayer(
        incoming = l_pool2,
        p = 0.3,
        name = 'drop2',
        )
    l_fc1 = lasagne.layers.DenseLayer(
        incoming = l_drop2,
        num_units = 128, # previously 128
        W = lasagne.init.Normal(std=0.01),
        name =  'fc1'
        )
    l_drop3 = lasagne.layers.DropoutLayer(
        incoming = l_fc1,
        p = 0.4,
        name = 'drop3',
        )
    l_fc2 = lasagne.layers.DenseLayer(
        incoming = l_drop3,
        num_units = n_classes,
        W = lasagne.init.Normal(std = 0.01),
        nonlinearity = None,
        name = 'fc2'
        )
    return {'l_in':l_in, 'l_out':l_fc2}
