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

    def gather(self, Fe, x, F):
        """Gather F from the regular grid.
        Parameters
        ----------
        Fe : [N1,N2,N3] complex64
            Function at equally spaced frequencies.
        x : (K, 3) float32
            Non-uniform frequencies.
        F : (K, ) complex64
            Init values at the non-uniform frequencies.
        Returns
        -------
        F : (K, ) complex64
            Values at the non-uniform frequencies.
        """
        n = Fe.shape
        m = self.m
        
        # skip points that are too far from the global grid
        ell = (np.round(cp.array(n) * x) ).astype(np.int32)  # nearest grid to x
        cond_out = cp.where((ell[:, 0]+n[0]//2+m >= 0) *\
            (ell[:, 1]+n[1]//2+m >= 0) *\
            (ell[:, 2]+n[2]//2+m >= 0) *\
            (ell[:, 0]+n[0]//2-m < n[0]) *\
            (ell[:, 1]+n[1]//2-m < n[1]) *\
            (ell[:, 2]+n[2]//2-m < n[2]))[0]
        ell = ell[cond_out]
        x = x[cond_out]
        Fc = F[cond_out]
        
        # gathering over 3 axes
        for i0 in range(-m, m):
            id0 = (n[0]//2 + ell[:, 0] + i0)
            cond0 = (id0 >= 0)*(id0 < n[0])  # check index z
            for i1 in range(-m, m):
                id1 = (n[1]//2 + ell[:, 1] + i1)
                cond1 = (id1 >= 0)*(id1 < n[1])  # check index y
                for i2 in range(-m, m):
                    id2 = (n[2]//2 + ell[:, 2] + i2)
                    cond2 = (id2 >= 0)*(id2 < n[2])  # check index x
                    # take index inside the global grid
                    cond = cp.where(cond0*cond1*cond2)[0]
                    id0 = id0[cond]
                    id1 = id1[cond]
                    id2 = id2[cond]
                    # compute weights
                    delta0 = ((ell[cond, 0] + i0) / (n[0]) - x[cond, 0])**2
                    delta1 = ((ell[cond, 1] + i1) / (n[1]) - x[cond, 1])**2
                    delta2 = ((ell[cond, 2] + i2) / (n[2]) - x[cond, 2])**2
                    Fkernel = self.cons[0] * \
                        cp.exp(self.cons[1] * (delta0 + delta1 + delta2))
                    # gather
                    Fc[cond] += Fe[id0, id1, id2] * Fkernel
        F[cond_out] += Fc
        return F
        
    def compfft(self, f):
        """Compesantion for smearing, followed by FFT
        Parameters
        ----------
        f : [n] * 3 complex64
            Function at equally-spaced coordinates
        Return
        ------
        Fe : [2 * n] * 3 complex64
            Fourier transform at equally-spaced frequencies
        """

        fe = cp.zeros([2 * self.n] * 3, dtype="complex64")
        fe[self.n//2:3*self.n//2, self.n//2:3*self.n//2, self.n //
            2:3*self.n//2] = f / ((2 * self.n)**3 * self.kernel)
        Fe = util.checkerboard(cp.fft.fftn(
            util.checkerboard(fe)), inverse=True)
        return Fe

    def fftcomp(self, G):
        """FFT followed by compesantion for smearing
        Parameters
        ----------
        Fe : [2 * n] * 3 complex64
            Fourier transform at equally-spaced frequencies
        Return
        ------            
        f : [n] * 3 complex64
            Function at equally-spaced coordinates        
        """
        F = util.checkerboard(cp.fft.fftn(
            util.checkerboard(G)), inverse=True)
        F = F[self.n//2:3*self.n//2, self.n//2:3*self.n//2, self.n //
              2:3*self.n//2] / ((2 * self.n)**3 * self.kernel)
        return F

    # def _delta(self, l, i, x):
    #     return ((l + i).astype('float32') / (2 * self.n) - x)**2
