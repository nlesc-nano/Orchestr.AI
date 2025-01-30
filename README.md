# MLFF_QD
Machine Learning Force Fields for Quantum Dots platform.

## Installation
Some packages are requiered for running the platform code. Below we explain what is requiered and how to install it.

### Install SchNetPack with pip
The simplest way to install SchNetPack is through pip which will automatically get the source code from PyPI:

```bash
pip install schnetpack
```

### Install SchNetPack from source
One can also install the most recent code from their repository:

```bash
git clone https://github.com/atomistic-machine-learning/schnetpack.git
cd schnetpack
pip install .
```

### Visualization tools
SchNetPack supports multiple logging backends via PyTorch Lightning. The default logger is Tensorboard.

## Getting started
The current version of the platform is developped for being run in a cluster. Thus, in this repository one can find the necessary code, a bash script example for submitting jobs in a slurm queue system and an input file example.

This plaform is currently being subject of several changes. Thus, on the meanwhile, descriptions of the files will be included here so they can be used.

### Preprocessing Files
In the preprocessing folder, one can find files for preparing positions, forces and energies after DFT simulations for the training. This folder has its own readme.

### Machine Learning Files
1. input_preparation.py: This file reads the xyz coordinates and constructs the npz files requiered for the training. Currently, it is not automatic, as it does not read the input.yaml and it has the name of the coordinates hardcoded.
2. xyztonpz.py: Used by input_preparation.py. It could be integrated there and a units conversion procedure will be included.
3. input.yaml: Example of an input file with the tuneable parameters.
4. main.py: This is the main file for the training code. The main functions can be found in the utils folder.
5. inference.py: It predicts the desired values after the training.
6. training_inference.sh: Example of a bash file for running both the training and the inference in a slurm based cluster. The input generation should be included here (or ideally in the python code).

### Training locally
If an user wants to run locally the training code, one can do the following:
```bash
python training.py --config input_file.yaml
```
By default, if no input file is specified, the training code looks for a file called input.yaml.

### Preprocessing
Currently, we have scripts used for preparaing the xyz files in the useful formats after DFT calculations with CP2K.
