#!python
#cython: language_level=3 

# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
module = 'galaxy_survey'

ext_modules = cythonize(Extension(
    module,
    sources=[module+'.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=["-O3", "-ffast-math","-march=native"],
    #extra_link_args=['-fopenmp']
    ),
    compiler_directives={'language_level' : "3"}
)

setup(ext_modules=ext_modules)
