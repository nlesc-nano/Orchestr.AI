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

### Install DScribe
One can install DScribe through pip which will automatically get the latest stable release:

```bash
pip install dscribe
```

They also provide a conda package through conda-forge:


```bash
conda install -c conda-forge dscribe
```

One can find more information [here](https://singroup.github.io/dscribe/latest/install.html).

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

## Extract Training Metrics from TensorBoard Event Files

This script, `analysis/extract_metrics.py`,  extracts scalar training metrics from TensorBoard event files and saves them to a CSV file.

- **`-p/--path`**:  Path to the TensorBoard event file. **(Required)**.
- **`-o/--output_file`**: Provides the path to the CSV file containing the training metrics.
*   Prerequisites **Required Python Packages**:
    *   `tensorboard`
    You can install these using pip:
    ```bash
    pip install tensorboard
    ```
### Command-Line Usage:
To run the script use the following command:

```bash
python analysis/extract_metrics.py -p <event_file_path> [-o <output_file_name>]
```

## Plotting Training Metrics for SchNet and Nequip

The `analysis/plot.py` script allows you to visualize training progress for your models. It accepts several command-line options to control its behavior. Here’s what each option means:

- **`--platform`**: Specifies the model platform. Use either `schnet` or `nequip`.
- **`--file`**: Provides the path to the CSV file containing the training metrics.
- **`--cols`**: Sets the number of columns for the subplot grid (default is 2).
- **`--out`**: Defines the output file name for the saved plot. The name should end with `.png`.


### Plotting SchNet Results

To plot the results for SchNet, use the following command:
```bash
python analysis/plot.py --platform schnet --file "path/to/schnet_metrics.csv" --cols 2 --out schnet_plot.png
```
Replace "path/to/schnet_metrics.csv" with the actual path to your SchNet metrics CSV file. 

### Plotting Nequip Results

To plot the results for Nequip, use the following command:
```bash
python analysis/plot.py --platform nequip --file "path/to/nequip_metrics.csv" --cols 2 --out nequip_plot.png
```
Replace "path/to/nequip_metrics.csv" with the actual path to your Nequip metrics CSV file.

These commands will generate plots for the respective platforms and save them as PNG files in the current working directory.