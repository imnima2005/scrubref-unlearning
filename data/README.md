# Data Directory

This directory stores datasets used by the ScrubRef-Rep experiments.

Datasets are selected through the active experiment configuration. During notebook execution, the data-loading pipeline checks the configured dataset, downloads the required files if they are not already available, and stores them inside this directory.

## Dataset Selection

The dataset is determined by the `dataset` field in the experiment configuration.

For example:
```python
CFG.dataset = "cifar10"
or
CFG.dataset = "cifar100"
```
The available dataset names and their corresponding loading logic are defined in:

## Automatic Download Workflow

When a notebook starts loading a dataset, the following process is applied:

1. The notebook reads the selected dataset from the active configuration.
2. The data loader checks whether the dataset already exists in this directory.
3. If the dataset is missing, it downloads the required files automatically.
4. The downloaded files are extracted or prepared in the expected dataset-specific location.
5. Subsequent runs reuse the local files and do not download the dataset again.
6. The loaded data is converted into the format required by the training and evaluation pipeline.

This makes the experiment setup reproducible without requiring manual dataset preparation.

## Expected Directory Structure
```text
data/
├── README.md
├── cifar-10-batches-py/
├── cifar-100-python/
└── imagewoof/
```
The dataset directories are created automatically when required. Do not rename or manually move them unless the corresponding paths in src/data_loader.py are updated as well.

## Supported Datasets
The repository currently supports the datasets used in the paper:

- CIFAR-10
- CIFAR-100
- ImageWoof-320

Dataset-specific preprocessing, download URLs, archive extraction, image resizing, and input handling are implemented in the data-loading module.

## Version Control
Raw datasets should not be committed to Git because they can be large and may be distributed under their own licenses.

Only this README file and, if necessary, small metadata files should be tracked by Git. The dataset files are ignored through the repository’s .gitignore configuration.

After cloning the repository, run the relevant notebook. The required dataset will be downloaded automatically into this directory.