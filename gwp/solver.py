import numpy as np
import cupy as cp
from gwp import util
from gwp import usfft
import matplotlib.pyplot as plt
import dxchange


class Solver():

    def __init__(self, n, nangles, alpha, beta, eps):
        # init box parameters for covering the spectrum (see paper)
        nf_start = np.int32(np.log2(n/64)+0.5)
        K = 3*nf_start
        step = (nf_start+1)/(K-1)
        nf = 2**(nf_start-range(K)*step)
        if K == 0:
            nf = np.array([0.5])
            K = 1
        xi_cent = n/nf/(2*np.pi)
        # make sure multiplication by 2 of xi_cent gives int to have aligned boxes
        xi_cent = np.int32(xi_cent*2)/2

        lam1 = 4*np.log(2)*(xi_cent*alpha)*(xi_cent*alpha)
        lam2 = lam1/(beta*beta)
        lam3 = lam1/(beta*beta)
        lambda1 = np.pi**2/lam1
        lambda2 = np.pi**2/lam2
        lambda3 = np.pi**2/lam3

        # box sizes in the frequency domain
        L1 = 4*np.int32(np.round(np.sqrt(-np.log(eps)/lambda1)))
        L2 = 4*np.int32(np.round(np.sqrt(-np.log(eps)/lambda2)))
        L3 = 4*np.int32(np.round(np.sqrt(-np.log(eps)/lambda3)))
        boxshape = np.array([L3, L2, L1]).swapaxes(0, 1)
        fgridshape = np.array([2 * n] * 3)
        # init Lebedev's sphere to cover the spectrum
        leb = util.take_sphere(nangles)

        # box grid in space
        x = ((cp.arange(-L1[-1]//2, L1[-1]//2)))/2
        y = ((cp.arange(-L2[-1]//2, L2[-1]//2)))/2
        z = ((cp.arange(-L3[-1]//2, L3[-1]//2)))/2
        [z, y, x] = cp.meshgrid(z, y, x, indexing='ij')
        x = cp.array([z.flatten(), y.flatten(),
                      x.flatten()]).astype('float32').swapaxes(0, 1)

        # global grid in space
        xg = cp.arange(-n, n)/2
        yg = cp.arange(-n, n)/2
        zg = cp.arange(-n, n)/2
        [zg, yg, xg] = np.meshgrid(zg, yg, xg, indexing='ij')
        xg = cp.array([zg.flatten(), yg.flatten(),
                       xg.flatten()]).astype('float32').swapaxes(0, 1)

        # init gaussian wave-packet basis for each layer
        gwpf = [None]*K
        for k in range(K):
            xn = (cp.arange(-L1[k]/2, L1[k]/2))/2
            yn = (cp.arange(-L2[k]/2, L2[k]/2))/2
            zn = (cp.arange(-L3[k]/2, L3[k]/2))/2
            [zn, yn, xn] = cp.meshgrid(zn, yn, xn, indexing='ij')
            gwpf[k] = cp.exp(-lambda1[k]*xn*xn-lambda2[k] *
                             yn*yn-lambda3[k]*zn*zn)
            gwpf[k] = gwpf[k].astype('float32').flatten()

        # find grid index for extracting boxes on layers>1 (small ones)
        # from the box on the the last layer (large box)
        inds = [None]*K
        for k in range(K):
            xi_centk = np.int32(xi_cent[k]*2)
            xi_centK = np.int32(xi_cent[-1]*2)
            xst = xi_centk-L1[k]//2+L1[-1]//2-xi_centK
            yst = -L2[k]//2+L2[-1]//2
            zst = -L3[k]//2+L3[-1]//2
            indsz, indsy, indsx = cp.meshgrid(cp.arange(zst, zst+L3[k]),
                                              cp.arange(yst, yst+L2[k]),
                                              cp.arange(xst, xst+L1[k]))
            inds[k] = (indsx+indsy*L1[-1]+indsz*L1[-1] *
                       L2[-1]).astype('int32').flatten()

        # 3d USFFT plan
        U = usfft.Usfft(n, eps)

        print('number of levels:', K)
        for k in range(K):
            print('box size on level', k, ':', boxshape[k])
            print('box center in space for level', k, ':', xi_cent[k])
            print('box center in frequency for level', k, ':', 2*xi_cent[k])

        self.U = U
        self.nangles = nangles
        self.boxshape = boxshape
        self.fgridshape = fgridshape        
        self.K = K
        self.leb = leb
        self.x = x
        self.xg = xg
        self.gwpf = gwpf
        self.inds = inds
        self.xi_cent = xi_cent

    def fwd(self, f):
        """Forward operator for GWP decomposition
        """
        # compensate for the USFFT kernel function in the space domain and apply 3D FFT
        F = self.U.compfft(cp.array(f))

        # find coefficients for each angle
        coeffs = [None]*self.K
        for k in range(self.K):
            coeffs[k] = np.zeros(
                [self.nangles, *self.boxshape[k]], dtype='complex64')
        for ang in range(1):
            print('angle', ang)
            # shift and rotate box on the last layer
            xr = self.x
            xr[:, 2] += self.xi_cent[-1]/2
            xr = util.rotate(xr, self.leb[ang])
            # switch to [-1/2,1/2) interval w.r.t. global grid
            xr /= cp.array(self.fgridshape/2)
            # gather value to the box grid on the last layer
            g = cp.zeros(int(np.prod(self.boxshape[-1])), dtype="complex64")
            g = self.U.gather(F, xr, g)
            dxchange.write_tiff(
                F.reshape(self.fgridshape).get().real, 'data/Ffwd')

            dxchange.write_tiff(
                g.reshape(self.boxshape[-1]).get().real, 'data/gr')

            # find coefficients on each box
            for k in range(self.K):
                # broadcast values to smaller boxes, multiply by the gwp kernel function
                fcoeffs = self.gwpf[k]*g[self.inds[k]]
                fcoeffs = fcoeffs.reshape(self.boxshape[k])
                # ifft on the box
                fcoeffs = util.checkerboard(cp.fft.ifftn(
                    util.checkerboard(fcoeffs), norm='ortho'), inverse=True)
                coeffs[k][ang] = fcoeffs.get()
        return coeffs

    def adj(self, coeffs):
        """Adjoint operator for GWP decomposition
        """
        # build spectrum by using gwp coefficients
        F = cp.zeros(int(np.prod(self.fgridshape)), dtype="complex64")
        for ang in range(1):
            print('angle', ang)
            g = cp.zeros(int(np.prod(self.boxshape[-1])), dtype='complex64')
            for k in range(self.K):
                fcoeffs = cp.array(coeffs[k][ang])
                # fft on the box
                fcoeffs = util.checkerboard(cp.fft.fftn(
                    util.checkerboard(fcoeffs), norm='ortho'), inverse=True)
                # broadcast values to smaller boxes, multiply by the gwp kernel function
                g[self.inds[k]] += self.gwpf[k]*fcoeffs.flatten()
            g = g.reshape(self.boxshape[-1])
            # rotate and shift box on the last layer
            xr = self.xg
            xr = util.rotate(xr, self.leb[ang], reverse=True)
            xr[:, 2] -= self.xi_cent[-1]
            # switch to [-1/2,1/2) interval w.r.t. box
            xr /= cp.array(self.boxshape[k]/2)
            # gather values from the box grid to the global grid
            F = self.U.gather(g, xr, F)  # sign change for adjoint FFT
        # apply 3D IFFT, compensate for the USFFT kernel function in the space domain
        dxchange.write_tiff(
            F.reshape(self.fgridshape).get().real, 'data/Ffwd')

        f = self.U.ifftcomp(
            F.reshape(self.fgridshape)).get().astype('complex64')

        return f
