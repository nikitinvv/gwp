import numpy as np
import cupy as cp
import os
import struct


def take_sphere(nangles):
    tmp = 0
    seek = 32*4
    with open(os.path.join(os.path.dirname(__file__), 'leb.bin'), 'rb') as fid:
        while (tmp < nangles):
            seek += tmp*3*4
            tmp = struct.unpack('i', fid.read(4))[0]
        if(tmp != nangles):
            print('Bad number of angles, a close possible number is ', tmp)
            exit()
        fid.seek(seek)
        leb = np.zeros([nangles, 3], dtype='float32')
        leb[:, 0] = struct.unpack(nangles*'f', fid.read(4*nangles))
        leb[:, 1] = struct.unpack(nangles*'f', fid.read(4*nangles))
        leb[:, 2] = struct.unpack(nangles*'f', fid.read(4*nangles))

    return leb


def rotate(x, leb, reverse=False):
    """Rotate coordinates with respect to a point on Lebedev's sphere
    """
    phi = np.arctan2(-leb[0], leb[1])
    R = np.array([[leb[0], np.cos(phi), -leb[2]*np.sin(phi)],
                  [leb[1], np.sin(phi),  leb[2]*np.cos(phi)],
                  [leb[2], 0.0, leb[0]*np.sin(phi)-leb[1]*np.cos(phi)]])
    if(reverse):
        R = R.swapaxes(0, 1)
    xr = cp.zeros(x.shape, dtype='float32')
    xr[:, 2] = R[0, 0]*x[:, 2] + R[0, 1]*x[:, 1] + R[0, 2]*x[:, 0]
    xr[:, 1] = R[1, 0]*x[:, 2] + R[1, 1]*x[:, 1] + R[1, 2]*x[:, 0]
    xr[:, 0] = R[2, 0]*x[:, 2] + R[2, 1]*x[:, 1] + R[2, 2]*x[:, 0]
    return xr


def checkerboard(array, inverse=False):
    """In-place FFTshift for even sized grids only.
    If and only if the dimensions of `array` are even numbers, flipping the
    signs of input signal in an alternating pattern before an FFT is equivalent
    to shifting the zero-frequency component to the center of the spectrum
    before the FFT.
    """
    def g(x):
        return 1 - 2 * (x % 2)

    for i in range(3):
        array = cp.moveaxis(array, i, -1)
        array *= g(cp.arange(array.shape[-1]) + 1)
        if inverse:
            array *= g(array.shape[-1] // 2)
        array = cp.moveaxis(array, -1, i)
    return array
