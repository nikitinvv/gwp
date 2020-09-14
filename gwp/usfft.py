import cupy as cp
import numpy as np
from gwp import util

class Usfft():
    """Provides unequally-spaced fast fourier transforms (USFFT).
    The USFFT, NUFFT, or NFFT is a fast-fourier transform from an uniform domain to
    a non-uniform domain or vice-versa. This module provides forward Fourier
    transforms for those two cases. The inverse Fourier transforms may be created
    by negating the frequencies on the non-uniform grid. 
    """

    def __init__(self, n, eps):
        # parameters for the USFFT transform
        mu = -np.log(eps) / (2 * n**2)
        Te = 1 / np.pi * np.sqrt(-mu * np.log(eps) + (mu * n)**2 / 4)
        m = np.int(np.ceil(2 * n * Te))
        # smearing kernel
        xeq = cp.mgrid[-n//2:n//2, -n//2:n//2, -n//2:n//2]
        kernel = cp.exp(-mu * cp.sum(xeq**2, axis=0)).astype('float32')
        # smearing constants
        cons = [np.sqrt(np.pi / mu)**3, -np.pi**2 / mu]

        self.n = n
        self.mu = mu
        self.m = m
        self.kernel = kernel
        self.cons = cons

    def gather(self, Fe, x, l):
        """Gather F from the regular grid Fe.
        Parameters
        ----------
        Fe : [l[0],l[1],l[2]] complex64
            Function at equally spaced frequencies.
        x : (K, 3) float32
            Non-uniform frequencies.
        l : (3, ) int32
            Shape of the global grid.
        Returns
        -------
        F : (K, ) complex64
            Values at the non-uniform frequencies.
        """
        # nearest grid to x
        ell = (np.round(cp.array(l) * x) ).astype(np.int32) 
        F = cp.zeros(x.shape[0],dtype='complex64')
        # gathering over 3 axes
        for i0 in range(-self.m, self.m):
            id0 = l[0]//2 + ell[:, 0] + i0         
            # check index z           
            cond0 = (id0 >= 0)*(id0 < l[0]) 
            for i1 in range(-self.m, self.m):
                id1 = l[1]//2 + ell[:, 1] + i1
                # check index y
                cond1 = (id1 >= 0)*(id1 < l[1])  
                for i2 in range(-self.m, self.m):
                    id2 = l[2]//2 + ell[:, 2] + i2
                    # check index x
                    cond2 = (id2 >= 0)*(id2 < l[2])  
                    # take index inside the global grid
                    cond = cp.where(cond0*cond1*cond2)[0]
                    # cond = cp.arange(ell.shape[0])
                    # compute weights
                    delta0 = ((ell[cond, 0] + i0) / (l[0]) - x[cond, 0])
                    delta1 = ((ell[cond, 1] + i1) / (l[1]) - x[cond, 1])
                    delta2 = ((ell[cond, 2] + i2) / (l[2]) - x[cond, 2])
                    #compensate for the grid change (2n->l)
                    delta0 *= (l[0]/(2*self.n))
                    delta1 *= (l[1]/(2*self.n))
                    delta2 *= (l[2]/(2*self.n))
                    
                    Fkernel = self.cons[0] * \
                        cp.exp(self.cons[1] * (delta0**2 + delta1**2 + delta2**2))
                    # gather          
                    F[cond] += Fkernel*Fe[id0[cond], id1[cond], id2[cond]]* Fkernel
        return F
        
    def compfft(self, f):
        """Compesantion for smearing, followed by inplace FFT
        Parameters
        ----------
        f : [n] * 3 complex64
            Function at equally-spaced coordinates
        Return
        ------
        fe : [2 * n] * 3 complex64
            Fourier transform at equally-spaced frequencies
        """

        fe = cp.zeros([2 * self.n] * 3, dtype="complex64")
        fe[self.n//2:3*self.n//2, self.n//2:3*self.n//2, self.n //
            2:3*self.n//2] = f / ((2 * self.n)**3 * self.kernel)
        fe = util.checkerboard(cp.fft.fftn(
            util.checkerboard(fe), norm='ortho'), inverse=True)
        return fe

    def ifftcomp(self, F):
        """Inplace FFT followed by compesantion for smearing
        Parameters
        ----------
        F : [2 * n] * 3 complex64
            Fourier transform at equally-spaced frequencies
        Return
        ------            
        F : [n] * 3 complex64
            Function at equally-spaced coordinates        
        """
        F = util.checkerboard(cp.fft.ifftn(
            util.checkerboard(F), norm='ortho'), inverse=True)
        F = F[self.n//2:3*self.n//2, self.n//2:3*self.n//2, self.n //
              2:3*self.n//2] / ((2 * self.n)**3 * self.kernel)
        return F