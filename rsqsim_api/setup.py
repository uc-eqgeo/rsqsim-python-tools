from setuptools import setup

setup(name='rsqsim-api',
      version='0.1',
      description='Read and write of RSQSim inputs and outputs',
      author='Andy Howell and several others hopefully',
      author_email='a.howell@gns.cri.nz',
      packages=['rsqsim_api'],
      install_requires=['numpy>=1.18.1'],
      )
