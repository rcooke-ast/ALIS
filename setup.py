#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
#
# Standard imports
#
import glob, os
from distutils.extension import Extension
#
# setuptools' sdist command ignores MANIFEST.in
#
#from distutils.command.sdist import sdist as DistutilsSdist
from setuptools import setup, find_packages
#
# Begin setup
#
setup_keywords = dict()
#
# THESE SETTINGS NEED TO BE CHANGED FOR EVERY PRODUCT.
#
setup_keywords['name'] = 'alis'
setup_keywords['description'] = 'ALIS'
setup_keywords['author'] = 'Ryan Cooke'
setup_keywords['author_email'] = 'rcooke.ast@gmail.com'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/rcooke-ast/ALIS'
#
# END OF SETTINGS THAT NEED TO BE CHANGED.
#
setup_keywords['version'] = '0.1.dev0' #get_version(setup_keywords['name'])
#
# Use README.rst as long_description.
#
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
#
# Set other keywords for the setup function.  These are automated, & should
# be left alone unless you are an expert.
#
# Treat everything in bin/ except *.rst as a script to be installed.
#
if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
        if not os.path.basename(fname).endswith('.rst')]
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['requires'] = ['Python (>2.7.0)']
# setup_keywords['install_requires'] = ['Python (>2.7.0)']
setup_keywords['zip_safe'] = False
setup_keywords['use_2to3'] = True
setup_keywords['packages'] = find_packages()
"""
setup_keywords['setup_requires']=['pytest-runner']
setup_keywords['tests_require']=['pytest']
"""

# Cython
# import numpy, os
# from Cython.Distutils import build_ext
# from Cython.Build import cythonize
# from distutils.extension import Extension

"""
include_gsl_dir = os.getenv('GSL_PATH')+'/include/'
lib_gsl_dir = os.getenv('GSL_PATH')+'/lib/'
pyx_files = glob.glob('alis/*.pyx')
setup_keywords['ext_modules']=[]
for pyx_file in pyx_files:
    pyx_split = pyx_file.split('.')
    pyx_split2 = pyx_split[0].split('/')
    # Generate Extension
    #ext = Extension(pyx_split2[1], [pyx_file],
    ext = Extension('pypit.'+pyx_split2[1], [pyx_file],
        include_dirs=[numpy.get_include(),
                    include_gsl_dir],
        library_dirs=[lib_gsl_dir],
        libraries=["gsl","gslcblas"]
    )
    # Append
    setup_keywords['ext_modules'].append(ext)
#for pyx_file in pyx_files:
#    pyx_split = pyx_file.split('/')
#    ext = cythonize(pyx_split[1])
#    setup_keywords['ext_modules'].append(ext)

setup_keywords['cmdclass']={'build_ext': build_ext}
"""

#
# Run setup command.
#
setup(**setup_keywords)

