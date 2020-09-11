import cupy as xp # gpu version
# import numpy as xp # cpu version

class Usfft():
    """Provides unequally-spaced fast fourier transforms (USFFT).
    The USFFT, NUFFT, or NFFT is a fast-fourier transform from an uniform domain to
    a non-uniform domain or vice-versa. This module provides forward Fourier
    transforms for those two cases. The inverse Fourier transforms may be created
    by negating the frequencies on the non-uniform grid. 
    """
    
    def __init__(self, n, eps):
        # parameters for the USFFT transform
        mu = -xp.log(eps) / (2 * n**2)
        Te = 1 / xp.pi * xp.sqrt(-mu * xp.log(eps) + (mu * n)**2 / 4)
        m = xp.int(xp.ceil(2 * n * Te))
        # smearing kernel 
        xeq = xp.mgrid[-n//2:n//2, -n//2:n//2, -n//2:n//2]
        kernel = xp.exp(-mu * xp.sum(xeq**2, axis=0)).astype('float32')
        # smearing constants
        cons = [xp.sqrt(xp.pi / mu)**3, -xp.pi**2 / mu]

        self.n = n
        self.mu = mu
        self.m = m
        self.kernel = kernel
        self.cons = cons
            
    def gather(self, Fe, x):
        """Gather F from the regular grid.
        Parameters
        ----------
        Fe : [2 * n] * 3 complex64
            Function at equally spaced frequencies.
        x : (N, 3) float32
            Non-uniform frequencies.
        Returns
        -------
        F : (N, ) complex64
            Values at the non-uniform frequencies.
        """        
        
        F = xp.zeros(x.shape[0], dtype="complex64")
        ell = ((2 * self.n * x) // 1).astype(xp.int32)  # nearest grid to x
        for i0 in range(-self.m, self.m):
            delta0 = self._delta(ell[:, 0], i0, x[:, 0])
            for i1 in range(-self.m, self.m):
                delta1 = self._delta(ell[:, 1], i1, x[:, 1])
                for i2 in range(-self.m, self.m):
                    delta2 = self._delta(ell[:, 2], i2, x[:, 2])
                    Fkernel = self.cons[0] * xp.exp(self.cons[1] * (delta0 + delta1 + delta2))
                    F += Fe[(self.n + ell[:, 0] + i0) % (2 * self.n),
                            (self.n + ell[:, 1] + i1) % (2 * self.n),
                            (self.n + ell[:, 2] + i2) % (2 * self.n)] * Fkernel
        return F
    
    def scatter(self, f, x, G):
        """Scatter f to the regular grid.
        Parameters
        ----------
        f : (N, ) complex64
            Values at non-uniform frequencies.
        x : (N, 3) float32
            Non-uniform frequencies.
        G : [2 * n] * 3 complex64
            Init function at equally spaced frequencies.
        Return
        ------
            G : [2 * n] * 3 complex64
            Function at equally spaced frequencies.
        """

        ell = ((2 * self.n * x) // 1).astype(xp.int32)  # nearest grid to x
        stride = ((2 * self.n)**2, 2 * self.n)
        for i0 in range(-self.m, self.m):
            delta0 = self._delta(ell[:, 0], i0, x[:, 0])
            for i1 in range(-self.m, self.m):
                delta1 = self._delta(ell[:, 1], i1, x[:, 1])
                for i2 in range(-self.m, self.m):
                    delta2 = self._delta(ell[:, 2], i2, x[:, 2])
                    Fkernel = self.cons[0] * xp.exp(self.cons[1] * (delta0 + delta1 + delta2))
                    ids = (           ((self.n + ell[:, 2] + i2) % (2 * self.n))
                        + stride[1] * ((self.n + ell[:, 1] + i1) % (2 * self.n))
                        + stride[0] * ((self.n + ell[:, 0] + i0) % (2 * self.n))
                    )  # yapf: disable
                    vals = f * Fkernel
                    # accumulate by indexes (with possible index intersections),
                    # TODO acceleration of bincount!!
                    vals = (xp.bincount(ids, weights=vals.real) +
                            1j * xp.bincount(ids, weights=vals.imag))
                    ids = xp.nonzero(vals)[0]
                    G[ids] += vals[ids]
        return G

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

        fe = xp.zeros([2 * self.n] * 3, dtype="complex64")
        fe[self.n//2:3*self.n//2, self.n//2:3*self.n//2, self.n//2:3*self.n//2] = f / ((2 * self.n)**3 * self.kernel)
        Fe = self.checkerboard(xp, xp.fft.fftn(self.checkerboard(xp, fe)), inverse=True)
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

        F = self.checkerboard(xp, xp.fft.fftn(self.checkerboard(xp, G)), inverse=True)
        F = F[self.n//2:3*self.n//2, self.n//2:3*self.n//2, self.n//2:3*self.n//2] / ((2 * self.n)**3 * self.kernel)
        return F

    def checkerboard(self, xp, array, inverse=False):
        """In-place FFTshift for even sized grids only.
        If and only if the dimensions of `array` are even numbers, flipping the
        signs of input signal in an alternating pattern before an FFT is equivalent
        to shifting the zero-frequency component to the center of the spectrum
        before the FFT.
        """
        def g(x):
            return 1 - 2 * (x % 2)

        for i in range(3):
            if array.shape[i] % 2 != 0:
                raise ValueError(
                    "Can only use checkerboard algorithm for even dimensions. "
                    f"This dimension is {array.shape[i]}.")
            array = xp.moveaxis(array, i, -1)
            array *= g(xp.arange(array.shape[-1]) + 1)
            if inverse:
                array *= g(array.shape[-1] // 2)
            array = xp.moveaxis(array, -1, i)
        return array

    def _delta(self, l, i, x):
        return ((l + i).astype('float32') / (2 * self.n) - x)**2


  
