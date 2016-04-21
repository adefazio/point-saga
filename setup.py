from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

fast_opts = ["-mtune=native", "-march=native", "-O3", 
                "-ftree-vectorize", "-msse2", "-msse3", "-fPIC", "-ffast-math", 
                 "-msse", "-mfpmath=sse", "-Wno-unused-function"]
#copts = "-std=c99",

setup(
    cmdclass = {'build_ext': build_ext},
    include_dirs = [np.get_include()],
    ext_modules = [
        Extension("fast_sampler", ["fast_sampler.pyx"], #"random_fast.cpp"
            language="c++",
            libraries=["stdc++"],
            include_dirs=[np.get_include()],
            extra_compile_args=fast_opts),
		Extension("sparse_util", ["optimization/sparse_util.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts),
		Extension("pegasos", ["optimization/pegasos.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts),
		Extension("saga", ["optimization/saga.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts),
		Extension("lsaga", ["optimization/lsaga.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts),
		Extension("wsaga", ["optimization/wsaga.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
            libraries=["stdc++"],
		    extra_compile_args=fast_opts),
		Extension("pointsaga", ["optimization/pointsaga.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts),
		Extension("sdca", ["optimization/sdca.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts),
		Extension("csdca", ["optimization/csdca.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts),
		Extension("get_loss", ["optimization/get_loss.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts),
		Extension("loss", ["optimization/loss.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts),
		Extension("hinge_loss", ["optimization/hinge_loss.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts),
		Extension("logistic_loss", ["optimization/logistic_loss.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts),
		Extension("squared_loss", ["optimization/squared_loss.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts)
    ]
)

