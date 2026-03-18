#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="disksurf",
    version="1.2.8",
    author="Richard Teague",
    author_email="rteague@mit.edu",
    description=("Infer and reproject a disk's 3D structure."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/richteague/disksurf",
    packages=["disksurf"],
    license="MIT",
    install_requires=[
        "scipy",
        "numpy",
        "matplotlib",
        "gofish >= 1.4.1.post1",
        "astropy",
        "emcee",
        ],
    extras_require={
        "mcmc": ["corner"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
