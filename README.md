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

### Files
* inference_code.py: It predicts the desired values after the training.
* input.yaml: Example of an input file with the tuneable parameters.
* input_preparation.py: This file reads the xyz coordinates and constructs the npz files requiered for the training. Currently, it is not automatic, as it does not read the input.yaml and it has the name of the coordinates hardcoded.
* training_inference.sh
* training_model_code.py
* xyztonpz.py: Used by input_preparation.py. It could be integrated there and a units conversion procedure will be included.

### Postprocessing
