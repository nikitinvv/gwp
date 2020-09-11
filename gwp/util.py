import cupy as xp
import numpy as np
import os
import struct

def take_sphere(nangles):
    tmp = 0
    seek = 32*4
    with open(os.path.join(os.path.dirname(__file__), 'leb.bin'),'rb') as fid:            
        while (tmp < nangles):
            seek += tmp*3*4
            tmp = struct.unpack('i',fid.read(4))[0]
        if(tmp != nangles):
            print('Bad number of angles, a close possible number is ', tmp)
            exit()
        fid.seek(seek)
        leb = np.zeros([nangles,3], dtype='float32')
        leb[:,0] = struct.unpack(nangles*'f',fid.read(4*nangles))
        leb[:,1] = struct.unpack(nangles*'f',fid.read(4*nangles))
        leb[:,2] = struct.unpack(nangles*'f',fid.read(4*nangles))
    
    return leb

def rotate(x, leb):
    """Rotate coordinates with respect to a point on Lebedev's sphere
    """
    phi = np.arctan2(-leb[0],leb[1])
    R = np.array([[leb[0], xp.cos(phi), -leb[2]*xp.sin(phi)],\
                [leb[1], xp.sin(phi),  leb[2]*xp.cos(phi)], \
                [leb[2], 0.0, leb[0]*xp.sin(phi)-leb[1]*xp.cos(phi)]])        
    xr = xp.zeros(x.shape, dtype='float32')                
    xr[:,0] = R[0,0]*x[:,0] + R[0,1]*x[:,1] + R[0,2]*x[:,2]
    xr[:,1] = R[1,0]*x[:,0] + R[1,1]*x[:,1] + R[1,2]*x[:,2]
    xr[:,2] = R[2,0]*x[:,0] + R[2,1]*x[:,1] + R[2,2]*x[:,2]
    return xr