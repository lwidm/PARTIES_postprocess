# -- setup.py
from setuptools import setup, find_packages # type: ignore

setup(
    name="parties_postprocess",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "h5py",
        "natsort",
        "matplotlib",
        "scipy",
        "torch",
    ],
    author="Lukas Widmer",
    description="",
    # python_requires=">3.8"
)
