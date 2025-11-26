# MLFF_QD  
### Unified Platform for Machine-Learning Force Fields for Quantum Dots 🚀

**MLFF_QD** is a unified, modular, and engine‑agnostic framework for training state‑of‑the‑art machine learning force fields (MLFFs) for **quantum dots (QDs)**.  
It integrates multiple ML engines under a single interface:

✅ **SchNet**
✅ **PaiNN**
✅ **NequIP**
✅ **Allegro**
✅ **MACE**

### 1. Installation
For the installation of the MLFF_QD platform and all the required packages, we recommend to create a conda environment using Python 3.12. 
Details will be provided in the following sections.

#### Installation of the mlff_qd package ⚙️
To install the `mlff_qd` platform, clone the repository and set up the environment as follows:

#### 1.1 Clone the repository
```bash
git clone https://github.com/nlesc-nano/MLFF_QD.git
cd MLFF_QD
```
#### 1.2 Set up the Conda environment 🛠️
To set up the conda environment, use the provided `environment.yaml` file. Once activated, install the `mace-torch` package as recommended.

```bash
conda env create -f environment.yaml
conda activate mlff
pip install mace-torch==0.3.14
```
#### 1.3 Install the mlff_qd package
Finally, install the package in editable mode:
```bash
pip install -e .
```
------
## Getting started
The current version of the platform is developed for being run in a cluster. Thus, in this repository one can find the necessary code, a bash script example for submitting jobs in a slurm queue system and an input file example.

This platform is currently being subject of several changes. Thus, on the meanwhile, descriptions of the files will be included here so they can be used.

### 2. Preprocessing tools
An input file example for the preprocessing of the data can be found in `config_files/preprocessing/preprocess_config.yaml`. The initial data for being processed should be placed in a consistent way to the paths indicated in the input file. This preprocessing tool is used for preparing the xyz files in the useful formats after DFT calculations with CP2K.

By default, the preprocessing code assumes that the input file is `preprocess_config.yaml`. If that is the case, it can be run as:
```bash
python -m mlff_qd.preprocessing.generate_mlff_dataset
```

However, if an user wants to specify a different custom configuration file for the preprocessing, the code can be run as:
```bash
python -m mlff_qd.preprocessing.generate_mlff_dataset --config my_experiment.yaml
```

### 3. Training Guide

MLFF_QD supports **two** ways to train:


####  **A) Using Unified YAML (Recommended)**  
*(The file is typically named `input.yaml` inside config_files/training/.)*

This YAML contains:
- shared hyperparameters  
- dataset config  
- the name of the engine (`platform:`)  
- training/evaluation settings  

You control workflow using these **two main flags**:

#### ✔ `--only-generate`
Only generate engine-specific YAML + converted dataset.  
**Training will not run.**

#### ✔ `--train-after-generate`
Generate engine YAML → **then start training automatically**.

### **Priority Rule**
If both flags are given:
```
--only-generate takes priority ⇒ NO training begins.
```

#### 3.1 CLI Usage

The main entry:
```bash
python -m mlff_qd.training
```
Arguments:
```
--config                Path to unified YAML file (input.yaml)
--engine                Override engine name (optional)
--input                 Override input XYZ file
--only-generate         Only produce engine YAML
--train-after-generate  Produce YAML + immediately train
--benchmark             Run cross-engine benchmarks
--post-process          Summaries from benchmark results
```


#### 3.2 Recommended Workflow: Unified YAML

##### 3.2.1 Generate engine YAML and immediately train

```bash
python -m mlff_qd.training --config input.yaml --engine <engine_name> --train-after-generate
python -m mlff_qd.training --config input.yaml --engine schnet --train-after-generate
```

##### 3.2.2 Only generate engine YAML (no training)

```bash
python -m mlff_qd.training --config input.yaml --engine <engine_name> --only-generate
python -m mlff_qd.training --config input.yaml --engine schnet --only-generate
```

#### 3.3 Running on SLURM
The repo includes `running_files/run_training.sh`.


##### 3.3.1 Generate + Train

```bash
sbatch run_training.sh input.yaml --engine <engine_name> --train-after-generate
sbatch run_training.sh input.yaml --engine schnet --train-after-generate
```

##### 3.3.2 Only Generate
```bash
sbatch run_training.sh input.yaml --engine <engine_name> --only-generate
sbatch run_training.sh input.yaml --engine schnet --only-generate
```
Same applies for any engine:
```bash
sbatch run_training.sh input.yaml --engine nequip --train-after-generate
sbatch run_training.sh input.yaml --engine allegro --train-after-generate
```


#### 3.4  Using Engine-Specific YAML Files 
*(Example: `config_files/training/schnet.yaml`, `config_files/training/nequip.yaml`)*

**You do not need** to use `--only-generate` or `--train-after-generate`.

**Train directly**

```bash
python -m mlff_qd.training --config schnet.yaml --engine schnet
```
or SLURM:

```bash
sbatch run_training.sh schnet.yaml --engine schnet
```



#### 3.5 Override dataset path (optional)
But you can specify input data inside `input.yaml`.
```bash
python -m mlff_qd.training --config schnet.yaml --engine schnet --input data/new.xyz
```


#### 3.6 Summary Table

| Task | Recommended Command |
|------|---------------------|
| Generate engine YAML only | `python -m mlff_qd.training --config input.yaml --engine <engine_name> --only-generate` |
| Generate + Train | `python -m mlff_qd.training --config input.yaml --engine <engine_name> --train-after-generate` |
| Train using engine-specific YAML | `python -m mlff_qd.training --config <engine_name>.yaml --engine <engine_name>` |
| SLURM – Generate only | `sbatch run_training.sh input.yaml --engine <engine_name> --only-generate` |
| SLURM – Generate + Train | `sbatch run_training.sh input.yaml --engine <engine_name> --train-after-generate` |

---

### Inference code
After the training has finished, an user can run the inference code that generates the MLFF:
```bash
python -m mlff_qd.training.inference
```
By default, it will look for a input file called input.yaml. Thus, if an user wants to specify another input file, one can do the following:
```bash
python -m mlff_qd.training.inference --config input_file.yaml
```

After inference, if an user wants to use fine-tuning, that option is also available in the following way:
```bash
python -m mlff_qd.training.fine_tuning
```
If an input file different from the default one was used, the procedure is the following:
```bash
python -m mlff_qd.training.fine_tuning --config input_file.yaml
```

### Postprocessing
More details will be added in future versions, but the postprocessing code is run as:
```bash
python -m mlff_qd.postprocessing
```
The postprocessing part of the code, requires also to install the following packages: plotly, kneed.

## CLI Mode - Extract Training Metrics from TensorBoard Event Files
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

## CLI Mode - Plotting Training Metrics for SchNet and Nequip

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

## GUI Mode: Interactive Metrics Extraction and Plotting with Streamlit

The `analysis/app.py` script offers a Streamlit GUI to extract metrics from TensorBoard event files and visualize SchNet/NequIP training progress with static (Matplotlib, saveable) or interactive (Plotly, display-only) plots. 

####  Prerequisites:  -  `streamlit`, `plotly`
  ```bash
  pip install streamlit plotly
  ```
  
### Launching the GUI:
  ```bash
  streamlit run analysis/app.py
  ```
