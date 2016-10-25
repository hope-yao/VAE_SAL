
import sys
sys.path.append("../")
print(sys.version)

from voxnet import isovox
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda.blas import GpuCorr3dMM
import numpy as np

vis_mat = np.load('../../voxnet/scripts/weights.npz')
W1 = vis_mat['conv1.W']  # size(32,1,5,5,5)
W2 = vis_mat['conv2.W'] # size(32,32,3,3,3)

contiguous_W1 = gpu_contiguous(W1)
contiguous_W2 = gpu_contiguous(W2)
strides = (1,1,1)
pad = (2, 2, 2)
corr_mm_op = GpuCorr3dMM(subsample=strides, pad=pad)(contiguous_W2,contiguous_W1)
print(corr_mm_op)

size = 32
wviz = W1[:, 0, :, :, :]
for i in range(wviz.shape[0]):
    w = wviz[i,:,:,:]
    # centerize the plot
    fz = len(w)
    xd = np.zeros((size,size,size))
    pad = (size-fz)/2
    xd[pad:pad+fz,pad:pad+fz,pad:pad+fz] = w
    # only visualize the largest value
    t = 0.2
    xd[xd<t]=0
    # store as png
    iv = isovox.IsoVox()
    img = iv.render(xd, as_html=True, name='./filters/test'+str(i))

print('visualization done')

