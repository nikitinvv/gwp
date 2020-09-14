from gwp import solver
import numpy as np

n = 128
nangles = 6
alpha = 1
beta = 3
ep = 1e-2
cl = solver.Solver(n, nangles, alpha, beta, ep)
f = np.random.random([n, n, n]).astype('float32')
coeffs = cl.fwd(f)
fr = cl.adj(coeffs)

s0 = np.sum(f*np.conj(fr))
s1 = np.float32(0)
for k in range(len(coeffs)):
    s1 += np.sum(coeffs[k]*np.conj(coeffs[k]))

print('Adjoint test <fwd(f),fwd(f)> ?= <f,adj(fwd(f))> :', s0, '?=', s1)
