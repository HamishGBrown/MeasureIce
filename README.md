# py_multislice

![](cbed.png)

A software tool for measuring ice thickness on the fly in cryo-EM

GPU accelerated using 
[pytorch](https://pytorch.org/)

Ionization based off [Flexible Atomic Code (FAC)](https://github.com/flexible-atomic-code/fac).

# Installation

1. Clone or branch this repo and the py_multislice package into a directory on your computer. Using git bash:

    $ git clone https://github.com/HamishGBrown/Measureice.git
    $ git clone https://github.com/HamishGBrown/py_multislice.git

2. It's best to create a specific anaconda environment within which to run the 
utilities. Within your terminal (linux and mac) or Anaconda interpreter (Windows):

    $ conda create --name measureice
    $ conda activate measureice

3. The multislice library requires a version of the pytorch library before 1.8.0 eg. 1.7.0 (the py_multislice library is incompatible with changes to the Fast Fourier transform libraries made for version 1.8.0 and after, rectifying this is a work in progress...). See instructions [here](https://pytorch.org/get-started/previous-versions/). Since the multislice component of the simulations is not particularly intensive it is simpler just to use the cpuonly version of this library eg:

    $ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch

4. In your terminal or Anaconda interpreter navigate to measureice directory and run the pip install command:

    $ cd measureIce
    $ pip install .

# Running Measureice

1. First you'll need to generate your calibration data. You need to know your microscope's accelerating voltage and the size of the apertures (in diffraction plane units like inverse Angstrom or mrad, not micron, you can measure this with a diffraction standard like a gold grid). Run Generate_MeasureIce_calibration.py with the --help argument to get more information on how to use the utility.

Documentation can be found [here](https://hamishgbrown.github.io/py_multislice/pyms/), for demonstrations and walk throughs on common simulation types see the Jupyter Notebooks in the [Demos](Demos/) folder. The Notebook for STEM-EELS is still under construction.

# Bug-fixes and contributions

Message me, leave a bug report and fork the repo. All contributions are welcome.

# Acknowledgements

This work was a collaboration with [Prof. Eric Hanssen](https://findanexpert.unimelb.edu.au/profile/333629-eric-hanssen) for teaching me the ins and outs of pytorch and numerous other discussions on computing and electron microscopy. Code for reading ser files is .


