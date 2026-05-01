from setuptools import Extension, setup
from Cython.Build import cythonize

ext = Extension(
    name="chain_reaction_core",
    sources=["chain_reaction.pyx"],
    language="c++",
    extra_compile_args=["-O3", "-std=c++11"],
)

setup(
    name="chain_reaction_core",
    ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}),
)
