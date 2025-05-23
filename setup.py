import os
import re

# To use a consistent encoding
from codecs import open as copen

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with copen(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with copen(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = find_version("__version__.py")

test_deps = [
    "pytest",
    "pytest-cov",
]

extras = {
    'test': test_deps,
}

setup(
    name='cloud.domain.adaptation',
    version=__version__,
    description="Doors detection long term",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Michele Antonazzi",
    author_email="micheleantonazzi@gmail.com",
    # Choose your license
    license='Apache Licence 2.0',
    include_package_data=True,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    tests_require=test_deps,
    # Add here the package dependencies
    install_requires=[
        'munch',
        'tqdm',
        'pyyaml',
        'numpy',
        'imageio',
        'lz4',
        'opencv-python',
        'Pillow',
        'torch',
        'torchvision',
       ' torchmetrics==1.2.0',
        'scipy',
        #'open3d',
        'pypng',
        'omegaconf',
        'hydra-core',
        'wandb',
        'codetiming',
        'pytorch_lightning',
        'scikit-image'
        #'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git',
        #'torch_scatter'
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    extras_require=extras,
)