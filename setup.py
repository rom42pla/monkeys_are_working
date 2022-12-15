import os
from glob import glob
from os.path import splitext, basename

import setuptools

with open("README.md", "r", encoding="utf-8") as fp:
    long_description = fp.read()

# with open("requirements.txt", "r", encoding="utf-8") as fp:
#     requirements = [s for s in fp.read().split("\n") if s]

setuptools.setup(
    name="monkeys_are_working",
    version="0.0.1",
    author="Romeo Lanzino",
    author_email="romeo.lanzino@gmail.com",
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rom42pla/monkeys_are_working",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "joblib==1.2.0",
        "icecream==2.1.3",
        "tqdm==4.64.1",
        "numpy==1.23.5",
        "einops==0.6.0",
        "torch==1.13.0",
        "torchvision==0.14.0",
        "torchaudio==0.13.0",
        "pandas==1.5.2",
        "opencv-python==4.6.0.66",
    ],
)