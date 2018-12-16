import glob
import os

import torch

from setuptools import find_packages
from setuptools import setup

requirements = ['torch', 'torchvision']

setup(
    name='image_captioning',
    packages=find_packages(exclude=('configs', 'tests')),
    #install_requires=requirements
)