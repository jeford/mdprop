#!/usr/bin/env python

from distutils.core import setup

setup(name='mdprop',
      version='0.5.0',
      description='Molecular dynamics with flexible integrators',
      author='Jason Ford',
      author_email='jeford@stanford.edu',
      packages=['mdprop'],
      requirements=['scipy', 'numpy', 'h5py'],
     )
