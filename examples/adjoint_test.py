from gwp import solver
import numpy as np
import dxchange
n = 64
nangles = 14
alpha = 1
beta = 3
ep = 1e-3
cl = solver.Solver(n, nangles, alpha, beta, ep)

f = np.ones([n,n,n], dtype='float32')
coeffs = cl.fwd(f)
cshape = coeffs[0][0].shape
coeffs[0][:] = 0

coeffs[0][0,cshape[0]//2,cshape[1]//2,cshape[2]//2] = 1
fr = cl.adj(coeffs)
dxchange.write_tiff(fr.real, 'data/fr')            
            
gr = cl.fwd(fr)

