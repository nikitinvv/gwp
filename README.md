# gwp
Forward and adjoint operators for Gaussian wave-packet decomposition on GPU

## Installation
python setup.py install

## Dependencies
cupy - for GPU acceleration of linear algebra operations in iterative schemes.

dxchange - read/write tiff fiels

Install dependencies: conda install -c conda-forge cupy  dxchange
## Examples
See examples/:

adjoint_test.py - perform the adjoint test for the forward and adjoint GWP operators

one_gwp.py - construct one gwp for a given angle and box level by using the adjoint operator

many_gwp.py - construct many gwp in one image for given angles and box levels by using the adjoint operator

perf.py - timing for the forward and adjoint gwp operators

synthetic.py - example of synthetic data decomposition
