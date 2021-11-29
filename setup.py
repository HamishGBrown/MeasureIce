"""Setup script for measureice, ensure that all dependent packages are installed."""
from setuptools import setup, find_packages

setup(
    name="measureIce",
    version="0.1",
    description="A software tool for measuring ice thickness in cryo-EM",
    author="Hamish Brown",
    author_email="hgbrown@unimelb.edu.au",
    url="https://github.com/HamishGBrown/measureIce/",
    packages=find_packages(),
    install_requires=[
        "h5py >= 2.10",
        "ipython >= 4.0",
        "scipy >= 1.1",
        "matplotlib >= 3.0",
        "numpy",
        "Pillow >= 6.0",
        "torch < 1.8.0",
        "tqdm >= 4.48",
        "ase",
        "pypng",
        "pyqtgraph",
        "black",
        "PyQT5",
        "mrcfile"
    ],
)
