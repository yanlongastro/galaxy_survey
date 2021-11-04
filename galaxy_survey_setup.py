#!python
#cython: language_level=3 

# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
module = 'galaxy_survey'
setup(ext_modules = cythonize(Extension(
    module,
    sources=[module+'.pyx'],
    language='c',
    #language_level="3",
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=["-O3", "-ffast-math","-march=native", "-fopenmp"],
    extra_link_args=['-fopenmp']
)))
