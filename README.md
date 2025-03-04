# MLFF_QD
Machine Learning Force Fields for Quantum Dots platform.

## Installation
Some packages are required to be installed before starting using our MLFF_QD platform.

## Installation for the preprocessing tools
Some packages are requiered for running the preprocessing. Below we explain what is requiered and how to install it:

### Install PLAMS
The simplest way to install PLAMS is through pip which will automatically get the source code from PyPI:

```bash
pip install PLAMS
```

One can find more information [here](https://www.scm.com/doc/plams/started.html#installing-plams).

### Install Compound Attachment Tool (CAT), nano-CAT and auto-FOX
To install these packages we recommend to download the latest versions from their original repositories in the links below:

* [CAT](https://github.com/nlesc-nano/CAT);

* [nano-CAT](https://github.com/nlesc-nano/nano-CAT);

* [auto-FOX](https://github.com/nlesc-nano/auto-FOX).

Then, to install them one can do the following in each folder:
```bash
pip install .
```

### Other packages
Apart from usual python packages such as numpy, scipy, sklearn or yaml, one needs to install also periodictable:

```bash
pip install periodictable
```

## Installation for the training
Some packages are requiered for running the training. Below we explain what is requiered and how to install it.

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

## Installation of the mlff_qd package
One can install the platform using pip in the following way:
```bash
git clone https://github.com/nlesc-nano/MLFF_QD.git
cd MLFF_QD  
pip install -e .
```

## Getting started
The current version of the platform is developped for being run in a cluster. Thus, in this repository one can find the necessary code, a bash script example for submitting jobs in a slurm queue system and an input file example.

This plaform is currently being subject of several changes. Thus, on the meanwhile, descriptions of the files will be included here so they can be used.

### Preprocessing tools
The input file for the preprocessing of the data can be found in config/preprocess_config.yaml. The initial data for being processed should be placed in data/raw. This tool is used for preparaing the xyz files in the useful formats after DFT calculations with CP2K.

By default, the preprocessing code assumes that the input file is config/preprocess_config.yaml. If that is the case, it can be run as:
```bash
python -m mlff_qd.preprocessing.generate_mlff_dataset
```

However, if an user wants to specify a different custom configuration file for the preprocessing, the code can be run as:
```bash
python -m mlff_qd.preprocessing.generate_mlff_dataset --config config/my_experiment.yaml
```
## YAML Generation for SchNet and Nequip Training

Use the `generate_scripts.py` script to automatically create a YAML configuration file for training. The script requires two parameters: the platform (`-p`) and the output file (`-o`).

### Generate YAML for SchNet

To generate a configuration file for SchNet, run:

```bash
python generate_scripts.py -p schnet -o schnet.yaml
```

This command produces `schnet.yaml` with all necessary settings for training using SchNet.

### Generate YAML for Nequip

```bash
python generate_scripts.py -p nequip -o nequip.yaml
```

This generates `nequip.yaml`, which is pre-configured for Nequip training.

### Training locally
If an user wants to run locally the training code, one can do the following:
```bash
python training.py --config input_file.yaml
```
By default, if no input file is specified, the training code looks for a file called input.yaml.
