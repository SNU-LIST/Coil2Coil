from setuptools import setup, find_packages

setup(
    name="ssmd",
    version="1.0", 
    packages=find_packages(),
    entry_points={"console_scripts": ["ssmd = ssmd.__main__:start_cli"]},
    install_requires=[
        "nptyping",
        "h5py",
        "imagesize",
        "overrides",
        "colorlog",
        "colored_traceback",
        "torchvision",
        "tensorboard",
        "torch",
        "tqdm",
        "scipy"
    ],
)
