## run pip install cython==3.1.5 before installing this package

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        name="metatcr.integration.mnnpy._utils", 
        sources=["metatcr/integration/mnnpy/_utils.pyx"],
        include_dirs=[np.get_include(), "metatcr/integration/mnnpy"],
    ),
]

setup(

    name="metatcr",
    version="0.1.0",
    author="Miaozhe Huo",
    author_email="miaozhhuo2-c@my.cityu.edu.hk",
    
    description="MetaTCR: A Framework for Analyzing Batch Effects in TCR Repertoire Datasets",
    
    # URL for the project's homepage
    url="https://github.com/floretli/MetaTCR",
    
    # Automatically find all packages
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    
    # List of package dependencies.
    install_requires=[
        # "touch>=1.9.1,<2",  # Tested on version 1.9-1.13
        "tqdm",
        "scipy",
        "biopython>=1.79",
        "matplotlib",
        "pandas==1.5.1",
        "numpy==1.23.5", ## for seat
        "tape_proteins==0.5",
        "faiss-gpu==1.7.1",
        "GitPython==3.1.13",
        "googledrivedownloader>=0.4",
        "seaborn==0.12.1",
        "umap-learn",
        "configargparse"
    ],
    python_requires=">=3.8",
        
    # License of the package
    license="GPLv3",
)