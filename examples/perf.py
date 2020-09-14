from gwp import solver
import numpy as np
import time
#timing functions
def tic():
    #Homemade version of matlab tic and toc functions    
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
       return time.time() - startTime_for_tictoc
n = 256
nangles = 6
alpha = 1
beta = 3
ep = 1e-2
cl = solver.Solver(n, nangles, alpha, beta, ep)
f = np.random.random([n, n, n]).astype('float32')
tic()
coeffs = cl.fwd(f)
print('fwd time', toc())
tic()
fr = cl.adj(coeffs)
print('adj time', toc())

