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

### Files
* inference_code.py
* input.yaml
* input_preparation.py
* training_inference.sh
* training_model_code.py
* xyztonpz.py

### Postprocessing
