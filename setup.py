from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        'weighted_hamming',
        ['weighted_hamming.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='weighted_hamming',
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)