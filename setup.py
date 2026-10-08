import os

from setuptools import Extension, setup

extensions = []
if os.environ.get("PYFRIDGE_NO_EXT") != "1":
    import numpy
    from Cython.Build import cythonize

    extensions = cythonize(
        [Extension("pyfridge._cd_fast", ["src/pyfridge/_cd_fast.pyx"],
                   include_dirs=[numpy.get_include()], optional=True)],
        compiler_directives={"language_level": 3, "boundscheck": False,
                             "wraparound": False, "cdivision": True},
    )
setup(ext_modules=extensions)
