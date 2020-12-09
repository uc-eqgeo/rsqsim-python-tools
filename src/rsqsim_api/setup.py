from setuptools import setup, find_packages

setup(name='rsqsim-api',
      version='0.1',
      description='Read and write of RSQSim inputs and outputs',
      author='Andy Howell and several others hopefully',
      author_email='a.howell@gns.cri.nz',
      packages=find_packages(),
      package_data={'rsqsim_api': ['visualisation/data/coastline/*']},
      )
