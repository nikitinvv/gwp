import numpy as np
import cupy as xp
from gwp import util
from gwp import usfft


class Solver():
    """...
    """

    def __init__(self, n, nangles, alpha, beta, os, eps):
        """
        """
        # init box parameters for covering the spectrum (see paper)
        nf_start = np.int32(np.log2(n/64)+0.5)
        K = 3*nf_start
        step = (nf_start+1)/(K-1)
        nf = 2**(nf_start-range(K)*step)
        xi_cent = n/nf/(2*np.pi)
        lam1 = 4*np.log(2)*(xi_cent*alpha)*(xi_cent*alpha)
        lam2 = lam1/(beta*beta)
        lam3 = lam1/(beta*beta)
        lambda1 = np.pi**2/lam1
        lambda2 = np.pi**2/lam2
        lambda3 = np.pi**2/lam3
        K1 = np.round(np.sqrt(-np.log(eps)/lambda1))
        K2 = np.round(np.sqrt(-np.log(eps)/lambda2))
        K3 = np.round(np.sqrt(-np.log(eps)/lambda3))
        L1 = 4*np.int32(os/2.0*K1)
        L2 = 4*np.int32(os/2.0*K2)
        L3 = 4*np.int32(os/2.0*K3)
        xi_cent = np.int32(xi_cent*os)/os
        boxshape = np.array([L3, L2, L1]).swapaxes(0, 1)

        # init Lebedev's sphere to cover the spectrum
        leb = util.take_sphere(nangles)

        # box grid
        x = (np.arange(-L1[-1]//2, L1[-1]//2) + np.int32(xi_cent[-1]*os))/os
        y = (np.arange(-L2[-1]//2, L2[-1]//2))/os
        z = (np.arange(-L3[-1]//2, L3[-1]//2))/os
        [x, y, z] = np.meshgrid(x, y, z)
        x = xp.array([x.flatten(), y.flatten(), z.flatten()]).swapaxes(0, 1)

        # init gaussian wave-packet basis for each layer
        gwpf = [None]*K
        for k in range(K):
            xn = (-L1[k]/2+xp.int32(xi_cent[k]*os) +
                  xp.arange(L1[k]))/os-xi_cent[k]
            yn = (-L2[k]/2+xp.arange(L2[k]))/os
            zn = (-L3[k]/2+xp.arange(L3[k]))/os
            [xn, yn, zn] = xp.meshgrid(xn, yn, zn)
            gwpf[k] = xp.exp(-lambda1[k]*xn*xn-lambda2[k] *
                             yn*yn-lambda3[k]*zn*zn).flatten()

        # find grid index for extracting boxes on layers>1 (small ones) from the box on the the last layer (big box)
        inds = [None]*K
        for k in range(K):
            xi_centk = np.int32(xi_cent[k]*os)
            xi_centK = np.int32(xi_cent[-1]*os)
            xst = xi_centk-L1[k]//2+L1[-1]//2-xi_centK
            yst = -L2[k]//2+L2[-1]//2
            zst = -L3[k]//2+L3[-1]//2
            indsx, indsy, indsz = xp.mgrid[xst:xst +
                                           L1[k], yst:yst+L2[k], zst:zst+L3[k]]
            inds[k] = (indsx+indsy*L1[-1]+indsz*L1[-1]*L2[-1]).flatten()

        # 3d USFFT plan
        U = usfft.Usfft(n, eps)

        self.x = x
        self.U = U
        self.nangles = nangles
        self.n = n
        self.K = K
        self.leb = leb
        self.gwpf = gwpf
        self.inds = inds
        self.boxshape = boxshape

    def fwd(self, f):
        """Forward operator for GWP decomposition
        """
        # compensate for the USFFT kernel function in the space domain and apply 3D FFT
        F = self.U.compfft(xp.array(f))

        # find coefficients for each angle
        coeffs = [None]*self.K
        for ang in range(self.nangles):
            print('angle', ang)
            # rotate box on the last layer
            xr = util.rotate(self.x, self.leb[ang])
            # gather value to the box grid on the last layer
            g = self.U.gather(F, xr)
            # find coefficients on each box
            for k in range(self.K):
                # broadcast values to smaller boxes, multiply by the gwp kernel function
                fcoeffs = self.gwpf[k]*g[self.inds[k]]
                # ifft on the box
                coeffs[k] = self.U.checkerboard(xp, xp.fft.ifftn(
                    self.U.checkerboard(xp, fcoeffs.reshape(self.boxshape[k]))), inverse=True)
        return coeffs

    def adj(self, coeffs):
        """Adjoint operator for GWP decomposition
        """
        # build spectrum by using gwp coefficients
        F = xp.zeros([(2 * self.n)**3], dtype="complex64")
        for ang in range(self.nangles):
            print('angle', ang)
            print(int(np.prod(self.boxshape[-1])))
            g = xp.zeros(int(np.prod(self.boxshape[-1])), dtype='complex64')
            for k in range(self.K):
                # fft on the box
                fcoeffs = self.U.checkerboard(xp, xp.fft.fftn(self.U.checkerboard(
                    xp, coeffs[k].reshape(self.boxshape[k]))), inverse=True)
                # broadcast values to smaller boxes, multiply by the gwp kernel function
                g[self.inds[k]] += self.gwpf[k]*fcoeffs.flatten()
            # rotate box on the last layer
            xr = util.rotate(self.x, self.leb[ang])
            # scatter values from the box grid on the last layer
            print('scatter', g.shape, F.shape)
            F = self.U.scatter(g, -xr, F)  # sign change for adjoint FFT
        # apply 3D FFT, compensate for the USFFT kernel function in the space domain
        f = self.U.fftcomp(F.reshape([2 * self.n] * 3))
        return f
