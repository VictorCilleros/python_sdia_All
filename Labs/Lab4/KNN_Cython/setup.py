from distutils.core import setup
from distutils.extension import Extension
import numpy

from Cython.Build import cythonize


# setup(
#     ext_modules=cythonize(
#         ["KNN.pyx"], annotate=True, language_level="3"
#     ),
# )


extensions = [
    Extension("KNN", ["KNN.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name="KNN",
    ext_modules=cythonize(["KNN.pyx"], annotate=True, language_level="3"),
)
