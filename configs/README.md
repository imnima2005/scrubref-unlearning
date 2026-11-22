# Configuration Directory

This directory stores YAML configuration files for the ScrubRef-Rep experiments.

Each configuration file defines the dataset, model architecture, training hyperparameters, unlearning parameters, and paths required to run an experiment.

## Usage

In the notebooks, experiments are loaded directly from a configuration file using `ExperimentConfig`:
```python
CFG = ExperimentConfig.from_yaml(REPO_DIR / "configs" / "publication" / "cifar10.yaml")
CFG.run_mode = "publication"
CFG.apply_mode()
```
or ( fast_publication notebook for imagewoof )
```python
CFG = ExperimentConfig.from_yaml(REPO_DIR / "configs" / "fast_publication" / "imagewoof.yaml")
CFG.run_mode = "fast_publication"
CFG.apply_mode()
```

By changing the configuration filename (e.g., switching between cifar10.yaml, cifar100.yaml), the entire experiment pipeline adapts to the selected dataset and settings automatically.

## Modifying Experiments
To test different settings, datasets, or unlearning hyperparameters, update the corresponding YAML file or override the configuration attributes directly in the notebook before calling CFG.apply_mode()