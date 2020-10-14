from setuptools import setup

setup(name='rsqsim-api',
      version='0.1',
      description='Read and write of RSQSim inputs and outputs',
      author='Andy Howell and several others hopefully',
      author_email='a.howell@gns.cri.nz',
      packages=['rsqsim_api'],
      install_requires=["numpy>=1.18.1",
                        "geopandas>=0.6.1",
                        "ipyvolume>=0.5.2",
                        "jupyter-client>=6.1.7",
                        "matplotlib>=3.1.1"],
      )
