from setuptools import setup, find_packages

setup(name='triangular_faults',
      version='1.0',
      description='Slip inversions using triangular meshes',
      author='Andy Howell',
      author_email='a.howell@gns.cri.nz',
      packages=find_packages(),
      install_requires=["numpy>=1.17",
                          "geopandas>=0.6.1",
                          "netcdf4>=1.4.2",
                          "ipython>=7.9.0",
                          "scipy>=1.3.1",
                          "rasterio>=1.1.0"]
      )
