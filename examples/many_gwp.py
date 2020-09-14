from gwp import solver
import numpy as np
import dxchange

n = 128
nangles = 14
alpha = 1
beta = 3
ep = 1e-2

levels = [0, 1, 1, 2, 2, 2]
angs = [2, 3, 12, 7, 8, 0]

# init coefficients
cl = solver.Solver(n, nangles, alpha, beta, ep)
f = np.random.random([n, n, n]).astype('float32')
coeffs = cl.fwd(f)

# zero coefficients and set only the one as 1
for k in range(len(coeffs)):
    coeffs[k][:] = 0
for j in range(len(levels)):
    level = levels[j]
    ang = angs[j]
    cshape = np.array(coeffs[level][0].shape)
    ind = cshape//2+np.int32((np.random.random(3)-0.5)*cshape*0.25)
    coeffs[level][ang, ind[0], ind[1], ind[2]] = 1

# recover gwp
fr = cl.adj(coeffs)
# check coefficients
coeffsr = cl.fwd(fr)

rname = 'data/fr_many'
iname = 'data/fi_many'

dxchange.write_tiff(fr.real, rname)
dxchange.write_tiff(fr.imag, iname)

print('gwp real part saved in'+rname+'.tiff')
print('gwp imag part saved in'+iname+'.tiff')

print('init levels', levels)
print('init angles', angs)
for ang in range(nangles):
    for level in range(len(coeffs)):
        print('ang', ang, 'level', level, 'norm coeffs',
              np.linalg.norm(coeffsr[level][ang]))
