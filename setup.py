import os
from setuptools import setup, find_packages

setup_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="raifhack_ds",
    version="1.0",
    author="raifhack raifhack",
    description="raifhack",
    packages=find_packages(),
)