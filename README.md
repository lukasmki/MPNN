# MPNN

Message Passing Neural Network Implementation

## Installation

### Easy Installation with Conda

```bash
> conda env create -f environment.yml
> conda activate mpnn
> pip install -e .
```

### Manually install dependencies

```bash
> conda create -n mpnn python=3.8 numpy
> conda activate mpnn
> conda install cudatoolkit
> conda install pytorch pytorch-cuda -c pytorch -c nvidia
> pip install -e .
```
