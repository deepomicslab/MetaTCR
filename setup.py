"""Install with:

    pip install -e . --no-deps

The Cython extension metatcr/integration/mnnpy/_utils (used only by the MNN
integration path, not by encoding/clustering) ships as source plus a prebuilt
cp38 .so; rebuild it for the active Python only if/when you run MNN integration.
"""
from setuptools import setup, find_packages

setup(
    name="metatcr",
    version="2.0.0",
    author="Miaozhe Huo",
    author_email="miaozhhuo2-c@my.cityu.edu.hk",
    description="Framework for analyzing batch effects in TCR repertoire datasets",
    url="https://github.com/deepomicslab/MetaTCR",
    license="GPLv3",
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    package_data={"metatcr.integration.mnnpy": ["*.pyx", "*.c", "*.h", "*.so"]},
)
