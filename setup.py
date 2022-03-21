#!/usr/bin/env python
import setuptools
import os

os.chmod("run.py", 0o744)

setuptools.setup(
    name='neurSLS',
    version='1.0',
    url='https://github.com/DecodEPFL/neurSLS',
    license='CC-BY-4.0 License',
    author='Clara Galimberti',
    author_email='clara.galimberti@epfl.ch',
    description='Neural System Level Synthesis',
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.7.1',
                      'numpy>=1.18.1',
                      'matplotlib>=3.1.3'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.8',
)
