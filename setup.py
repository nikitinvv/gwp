from setuptools import setup, find_packages
setup(
    name='gwp',
    version=open('VERSION').read().strip(),
    author='Viktor Nikitin',
    url='https://github.com/nikitinvv/gwp',
    packages=find_packages(),
    include_package_data = True,
    #scripts=[''],
    description='Gaussian wave-packet decompositon',
    zip_safe=False,
)