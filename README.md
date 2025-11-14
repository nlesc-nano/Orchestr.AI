# MLFF_QD
## Machine Learning Force Fields for Quantum Dots platform. 🚀

**MLFF_QD** is a unified, modular, and engine‑agnostic framework for training state‑of‑the‑art machine learning force fields (MLFFs) for **quantum dots (QDs)**.  
It integrates multiple ML engines under a single interface:

✅ **SchNet**
✅ **PaiNN**
✅ **NequIP**
✅ **Allegro**
✅ **MACE**

## Installation
For the installation of the MLFF_QD platform and all the required packages, we recommend to create a conda environment using Python 3.12. 
Details will be provided in the following sections.

### Installation of the mlff_qd package ⚙️
To install the `mlff_qd` platform, clone the repository and set up the environment as follows:

####  Clone the repository
```bash
git clone https://github.com/nlesc-nano/MLFF_QD.git
cd MLFF_QD
```
#### Set up the Conda environment 🛠️
To set up the conda environment, use the provided `environment.yaml` file. Once activated, install the `mace-torch` package as recommended.

```bash
conda env create -f environment.yaml
conda activate mlff
pip install mace-torch==0.3.13
```
####  Install the mlff_qd package
Finally, install the package in editable mode:
```bash
pip install -e .
```
------
## Getting started
The current version of the platform is developped for being run in a cluster. Thus, in this repository one can find the necessary code, a bash script example for submitting jobs in a slurm queue system and an input file example.

This plaform is currently being subject of several changes. Thus, on the meanwhile, descriptions of the files will be included here so they can be used.

### Preprocessing tools
An input file example for the preprocessing of the data can be found in `config_files/preprocess_config.yaml`. The initial data for being processed should be placed in a consistent way to the paths indicated in the input file. This preprocessing tool is used for preparaing the xyz files in the useful formats after DFT calculations with CP2K.

By default, the preprocessing code assumes that the input file is `preprocess_config.yaml`. If that is the case, it can be run as:
```bash
python -m mlff_qd.preprocessing.generate_mlff_dataset
```

However, if an user wants to specify a different custom configuration file for the preprocessing, the code can be run as:
```bash
python -m mlff_qd.preprocessing.generate_mlff_dataset --config my_experiment.yaml
```

## Training Guide
To run the training code, on can use the following command, which by default looks for a config file named `input.yaml`:
```bash
python -m mlff_qd.training
```
In `config_files/` one can find an example of the file. Here, for any of the engines available in the platform, one can find the common parameters used for setting up the model and the training.

To specify a different config file, one should run the following command:
```bash
python -m mlff_qd.training --config nequip.yaml
```

---

This loads the unified.yaml config and optionally overrides the engine at runtime.
You can run any engine by changing `--engine` to one of:
```bash
schnet, painn, nequip, allegro, mace, fusion
```
We are showing examples with nequip, you can choose anyone. **The training process will automatically convert the data format according to the platform (engine) selected.**

#### 🟩 Commands Using Unified YAML (`unifiedYaml.yaml`)

> `unifiedYaml.yaml` should contain `platform: nequip` and optionally `input_xyz_file`.

| Use Case | Command | Notes |
|----------|---------|-------|
| Engine and input both defined in YAML | `sbatch run_idv.sh unifiedYaml.yaml` | `unifiedYaml.yaml` contains `platform: nequip` and `input_xyz_file` |
| Engine overridden via CLI, input from YAML | `sbatch run_idv.sh unifiedYaml.yaml --engine nequip` | Use if YAML has `input_xyz_file` but platform may vary |
| Engine and input both overridden via CLI | `sbatch run_idv.sh unifiedYaml.yaml --engine nequip --input ./basic.xyz` | Most flexible: ignores YAML settings |
| Engine overridden, input missing | `sbatch run_idv.sh unifiedYaml.yaml --engine nequip` | ❌ Will fail if `input_xyz_file` is missing from YAML |
| Input passed via `--input`, YAML has no dataset | `sbatch run_idv.sh unifiedYaml.yaml --engine nequip --input ./basic.xyz` | ✅ Safe way to inject input without editing YAML |

---

#### 🟦 Commands Using Engine-Specific YAML (e.g., `nequip.yaml`)

> These YAMLs should include both `platform: nequip` and `input_xyz_file`.

| Use Case | Command | Notes |
|----------|---------|-------|
| Use engine-specific YAML with `--engine` | `sbatch run_idv.sh nequip.yaml --engine nequip` | YAML must contain `input_xyz_file` |
| Override input in engine-specific YAML | `sbatch run_idv.sh nequip.yaml --engine nequip --input ./basic.xyz` | Use to test with alternate datasets |

## Additional Flags: `--only-generate` and `--train-after-generate`

Control **data/config generation** and **training** phases using these flags:
- `--only-generate`:  
  Only generate the engine-specific YAML and/or convert data, **without starting training**.
- `--train-after-generate`:  
  Generate data/config, then **immediately start training** using the generated engine YAML.

### Priority Rules:
If you provide **both** flags, `--only-generate` takes precedence and training will **not** start.

```bash
# Only generate engine YAML and converted data (no training)
sbatch run_idv.sh unifiedYaml.yaml --engine nequip --only-generate

# Generate and then train (run both steps)
sbatch run_idv.sh unifiedYaml.yaml --engine nequip --train-after-generate
```

---

## 📝 Note on Engine YAMLs

If you are **already using an engine-specific YAML** (e.g., `nequip.yaml`, `schnet.yaml`):

- You **do not need** to use `--only-generate` or `--train-after-generate`.
- Just run:
```bash
sbatch run_idv.sh nequip.yaml --engine nequip
```

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
If an user wants to use an input file different from the default config.yaml, the procedure is the following:
```bash
python -m mlff_qd.training.fine_tuning --config input_file.yaml
```
The postprocessing part of the code, requieres also to install the following packages: plotly, kneed.

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
