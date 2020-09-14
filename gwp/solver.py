import numpy as np
import cupy as cp
from gwp import util
from gwp import usfft


class Solver():
    """Provides forward and adjoint operators for Gaussian Wave-packet (GWP) decompositon on
    GPU with using cupy library. For details see http://www.mathnet.ru/links/f0cbda0c8155c9c4d0ff6dd015c9ec78/vmp881.pdf
    """

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
        xi_cent = np.int32(xi_cent*2)

        lam1 = 4*np.log(2)*(xi_cent/2*alpha)*(xi_cent/2*alpha)
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

        # box grid in frequency
        x = ((cp.arange(-L1[-1]//2, L1[-1]//2)))
        y = ((cp.arange(-L2[-1]//2, L2[-1]//2)))
        z = ((cp.arange(-L3[-1]//2, L3[-1]//2)))
        [z, y, x] = cp.meshgrid(z, y, x, indexing='ij')
        x = cp.array([z.flatten(), y.flatten(),
                      x.flatten()]).astype('float32').swapaxes(0, 1)

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
            xi_centk = np.int32(xi_cent[k])
            xi_centK = np.int32(xi_cent[-1])
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

        # find spectrum subregions for each rotated box
        subregion = np.zeros([nangles, 6], dtype='int32')
        for ang in range(nangles):
            xr = x.copy()
            xr[:, 2] += xi_cent[-1]
            xr = util.rotate(xr, leb[ang])
            ell = np.round(xr + cp.array(fgridshape)/2).astype(np.int32)
            # borders in z
            subregion[ang, 0] = cp.min(ell[:, 0])-U.m
            subregion[ang, 1] = cp.max(ell[:, 0])+U.m
            # borders in y
            subregion[ang, 2] = cp.min(ell[:, 1])-U.m
            subregion[ang, 3] = cp.max(ell[:, 1])+U.m
            # borders in x
            subregion[ang, 4] = cp.min(ell[:, 2])-U.m
            subregion[ang, 5] = cp.max(ell[:, 2])+U.m
            #print(ang, subregion[ang])

        print('number of levels:', K)
        for k in range(K):
            print('box size on level', k, ':', boxshape[k])
            print('box center in frequency for level', k, ':', xi_cent[k])
        self.U = U
        self.nangles = nangles
        self.boxshape = boxshape
        self.fgridshape = fgridshape
        self.K = K
        self.leb = leb
        self.x = x
        self.gwpf = gwpf
        self.inds = inds
        self.xi_cent = xi_cent
        self.subregion = subregion

    def subregion_ids_fwd(self, ang):
        """
        Calculate indices for the subregion in the global grid with wrapping 
        for the gathering operation in the forward transform
        Parameters
        ----------
        ang : float32
            Box orientation angle
        Returns
        -------        
        zg,yg,xg: float32
            z,y,x arrays of coordinates
        """
        # extract subregion
        zg = cp.arange(
            self.subregion[ang, 0], self.subregion[ang, 1]) % self.fgridshape[0]
        yg = cp.arange(
            self.subregion[ang, 2], self.subregion[ang, 3]) % self.fgridshape[1]
        xg = cp.arange(
            self.subregion[ang, 4], self.subregion[ang, 5]) % self.fgridshape[2]
        return zg, yg, xg

    def coordinates_fwd(self, ang):
        """
        Compute coordinates of the local box grid on the last layer 
        for the gathering operation in the adjoint transform
        Parameters
        ----------
        ang : float32
            Box orientation angle
        Returns
        -------        
        xr: [Ns,3]: float32
            (z,y,x) array of coordinates
        """
        # shift and rotate box on the last layer
        xr = self.x.copy()
        xr[:, 2] += self.xi_cent[-1]
        xr = util.rotate(xr, self.leb[ang])
        # move to the subregion
        xr[:, 0] -= self.subregion[ang, 0]
        xr[:, 1] -= self.subregion[ang, 2]
        xr[:, 2] -= self.subregion[ang, 4]
        # switch to [-1/2,1/2) interval w.r.t. global grid
        xr /= cp.array(self.fgridshape)
        return xr

    def subregion_ids_adj(self, ang):
        """
        Calculate indices for the subregion in the global grid with wrapping.
        Parameters
        ----------
        ang : float32
            Box orientation angle
        Returns
        -------        
        xg: float32
            1d flatten array of coordinates
        """        
        zg = cp.arange(self.subregion[ang, 0],
                       self.subregion[ang, 1]) % self.fgridshape[0]
        yg = cp.arange(self.subregion[ang, 2],
                       self.subregion[ang, 3]) % self.fgridshape[0]
        xg = cp.arange(self.subregion[ang, 4],
                       self.subregion[ang, 5]) % self.fgridshape[0]
        [zg, yg, xg] = cp.meshgrid(zg, yg, xg, indexing='ij')
        xg = cp.array([zg.flatten(), yg.flatten(),
                       xg.flatten()]).swapaxes(0, 1)
        xg = xg[:, 0]*self.fgridshape[1]*self.fgridshape[2] + \
            xg[:, 1]*self.fgridshape[2]+xg[:, 2]
        return xg

    def coordinates_adj(self, ang):
        """
        Compute coordinates of the global grid 
        for the gathering operation with respect to the box on the last layer.
        Parameters
        ----------
        ang : float32
            Box orientation angle
        Returns
        -------        
        xr: [Ns,3]: float32
            (z,y,x) array of coordinates
        """
        # form 1D array of indeces
        zg = cp.arange(self.subregion[ang, 0],
                       self.subregion[ang, 1])
        yg = cp.arange(self.subregion[ang, 2],
                       self.subregion[ang, 3])
        xg = cp.arange(self.subregion[ang, 4],
                       self.subregion[ang, 5])
        [zg, yg, xg] = cp.meshgrid(zg, yg, xg, indexing='ij')
        xg = cp.array([zg.flatten(), yg.flatten(),
                       xg.flatten()]).swapaxes(0, 1)
        # rotate and shift
        xr = (xg - cp.array(self.fgridshape)/2)
        xr = util.rotate(xr, self.leb[ang], reverse=True)
        xr[:, 2] -= self.xi_cent[-1]
        # switch to [-1/2,1/2) interval w.r.t. box
        xr /= cp.array(self.boxshape[-1])
        return xr

    def fwd(self, f):
        """Forward operator for GWP decomposition
        Parameters
        ----------
        f : [N,N,N] complex64
            3D function in the space domain
        Returns
        -------
        coeffs : [K](Nangles,L3,L2,L1) complex64
            Decomposition coefficients for box levels 0:K, angles 0:Nangles, 
            defined on box grids with sizes [L3[k],L2[k],L1[k]], k=0:K
        """
        print('fwd transform')
        # 1) Compensate for the USFFT kernel function in the space domain and apply 3D FFT
        F = self.U.compfft(cp.array(f))

        # allocate memory for coefficeints
        coeffs = [None]*self.K
        for k in range(self.K):
            coeffs[k] = np.zeros(
                [self.nangles, *self.boxshape[k]], dtype='complex64')

        # loop over box orientations
        for ang in range(0,self.nangles):
            print('angle', ang)
            # 2) Interpolation to the local box grid.
            # Gathering operation from the global to local grid.
            # extract ids of the global spectrum subregion contatining the box
            [idsz, idsy, idsx] = self.subregion_ids_fwd(ang)
            # calculate box coordinates in the space domain
            xr = self.coordinates_fwd(ang)
            # gather values to the box grid
            g = self.U.gather(F[cp.ix_(idsz, idsy, idsx)], xr, F.shape)            
            # 3) IFFTs on each box.
            # find coefficients on each box
            for k in range(self.K):
                # broadcast values to smaller boxes, multiply by the gwp kernel function
                fcoeffs = self.gwpf[k]*g[self.inds[k]]
                fcoeffs = fcoeffs.reshape(self.boxshape[k])
                # ifft on the box
                fcoeffs = util.checkerboard(cp.fft.ifftn(
                    util.checkerboard(fcoeffs), norm='ortho'), inverse=True)
                # normalize 
                fcoeffs /= (np.prod(self.boxshape[-1]))
                coeffs[k][ang] = fcoeffs.get()
        return coeffs

    def adj(self, coeffs):
        """Adjoint operator for GWP decomposition
        Parameters
        ----------
        coeffs : [K](Nangles,L3,L2,L1) complex64
            Decomposition coefficients for box levels 0:K, angles 0:Nangles, 
            defined box grids with sizes [L3[k],L2[k],L1[k]], k=0:K        
        Returns
        -------
        f : [N,N,N] complex64
            3D function in the space domain        
        """
        print('adj transform')

        # build spectrum by using gwp coefficients
        F = cp.zeros(int(np.prod(self.fgridshape)), dtype="complex64")
        # loop over box orientations
        for ang in range(0,self.nangles):        
            print('angle', ang)
            g = cp.zeros(int(np.prod(self.boxshape[-1])), dtype='complex64')
            for k in range(self.K):
                fcoeffs = cp.array(coeffs[k][ang])
                # normalize 
                fcoeffs /= (np.prod(self.boxshape[-1]))               
                # fft on the box
                fcoeffs = util.checkerboard(cp.fft.fftn(
                    util.checkerboard(fcoeffs), norm='ortho'), inverse=True)
                # broadcast values to smaller boxes, multiply by the gwp kernel function
                g[self.inds[k]] += self.gwpf[k]*fcoeffs.flatten()
            g = g.reshape(self.boxshape[-1])
            
            # 2) Interpolation to the global grid
            # Conventional scattering operation from the global to local grid is replaced
            # by an equivalent gathering operation.
            # calculate global grid coordinates in the space domain
            xr = self.coordinates_adj(ang)
            # extract ids of the global spectrum subregion contatining the box
            ids = self.subregion_ids_adj(ang)
            # gathering to the global grid
            F[ids] += self.U.gather(g, xr, g.shape)
        
        # 3) apply 3D IFFT, compensate for the USFFT kernel function in the space domain
        f = self.U.ifftcomp(
            F.reshape(self.fgridshape)).get().astype('complex64')

        return f
