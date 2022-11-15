"""
# Description:
#  Initial setup
#
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
"""

from setuptools import setup, find_packages

setup(
    name="denoise",
    version="1.0", 
    packages=find_packages(),
    entry_points={"console_scripts": ["module = module.__main__:start_cli"]},
    install_requires=[
        "torch==1.12",
        "torchvision==0.13",
        "tensorboard",
        "nptyping",
        "h5py",
        "imagesize",
        "overrides",
        "colorlog",
        "colored_traceback",
        "tqdm",
        "nibabel",
        "pydicom",
        "scipy",
        "mat73"
    ],
)
