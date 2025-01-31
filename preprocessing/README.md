## Installation
Some packages are requiered for running the preprocessing code. Below we explain what is requiered and how to install it:

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
