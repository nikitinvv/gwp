from gwp import solver
import numpy as np
n = 64
nangles = 6
alpha = 1
beta = 3
os = 2
ep = 1e-3 
cl = solver.Solver(n, nangles, alpha, beta, os, ep)

f = np.ones([n,n,n], dtype='float32')
coeffs = cl.fwd(f)
f1 = cl.adj(coeffs)