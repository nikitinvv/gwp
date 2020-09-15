from gwp import solver
import numpy as np
import struct
import dxchange

n = 128
alpha = 1
beta = 3
ep = 1e-2
nangles = 26
cl = solver.Solver(n, nangles, alpha, beta, ep)

f = dxchange.read_tiff('data/lens_srt_128-128-128.tiff')

coeffs = cl.fwd(f)
fr = cl.adj(coeffs)/nangles

dxchange.write_tiff(fr.real,'data/lens_srt_128-128-128_rec'+str(nangles))


