# ScrubRef-Rep

Data-free class unlearning through proxy synthesis, representational scrubbing, and retained-only refinement.

> **Status:** This README documents the repository workflow and the paper-reported results supplied with the project. The numerical results below are reported by the paper and have **not** been independently reproduced by this repository.

## Overview

ScrubRef-Rep is a data-free class-unlearning method. Its workflow has three phases:

1. **Proxy synthesis (Phase 0):** Starting from random noise, generative gradient ascent synthesizes a proxy set for the target class.
2. **Representation scrubbing (Phase 1):** Early teacher layers are frozen. The trainable feature tail and classifier head are updated using proxy degradation together with retained knowledge distillation (KD).
3. **Retained-only refinement (Phase 2):** Masked KD and MixUp refine the model using retained data and restore the retained decision boundary.

The repository supports experiments on **CIFAR-10**, **CIFAR-100**, and **ImageWoof-320**. The paper compares the method with retain-only retraining, simple controls, UNSIR, Bad Teacher, SCRUB, and SSD.

## Reported results

The following figures are reported in the paper's abstract and are included here as paper-reported claims, not as independently reproduced repository results:

- Forget accuracy: **0.00%**
- Retained utility: **89.18%, 70.70%, 92.30%**
- Membership-inference attack ROC-AUC: **0.505, 0.508, 0.527**
- Speed-up: **up to 22.7×**

The supplied information does not specify which values map to which dataset. Verify the dataset-specific mapping against the paper's table before making dataset-level claims.

## Repository layout

```text
.
├── readme.md
├── requirements.txt
├── License
├── CITATION.cff
├── .gitignore
├── notebooks/
│   ├── <publication notebook; numbered 01; exact filename to be confirmed>
│   └── <fast-publication notebook; exact filename to be confirmed>
├── configs/
│   ├── fast-publication/
│   │   └── imagewoof.yaml
│   └── publication/
│       ├── cifar10.yaml
│       └── cifar100.yaml
├── data/
│   └── <datasets downloaded at runtime by the notebooks>
├── checkpoints/
│   └── <original/base models>
├── results/
│   └── <created at runtime>
└── src/
    └── <project modules>
```

The exact notebook filenames are intentionally left as placeholders because they were not supplied. Replace the two notebook placeholders and the Colab links below with the repository's actual paths before publishing.

## Requirements and setup

From the repository root:

```bash
pip install -r requirements.txt
```

The notebooks download the datasets into `data/` when run. Original/base models are stored in `checkpoints/`, and runtime outputs are written to `results/`.

## Notebooks

Each notebook is intended to clone the repository in its first cell before running the experiment. Open the corresponding notebook in Colab using the placeholders below.

### Publication notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imnima2005/scrubref-unlearning/blob/main/notebooks/01_multi_seed_benchmarking%20(4).ipynb)

The publication notebook is the notebook numbered **01**. It covers CIFAR-10 and CIFAR-100 and is less conservative about RAM than the fast-publication notebook. Select the desired configuration in the notebook; that selection is applied to `ExperimentConfig`.

Available publication configurations:

- `configs/publication/cifar10.yaml`
- `configs/publication/cifar100.yaml`

### Fast-publication notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imnima2005/scrubref-unlearning/blob/main/notebooks/02_fast_publication_imagewoof_(3)%20(1).ipynb)

The fast-publication notebook is more conservative about RAM. It is currently defaulted to ImageWoof-320 and uses:

- `configs/fast-publication/imagewoof.yaml`

## Running an experiment

1. Clone this repository, or open one of the notebooks in Colab. The first notebook cell clones the repository.
2. Open the relevant notebook.
3. Choose the dataset/configuration supported by that notebook.
4. Confirm that the selected YAML configuration is applied to `ExperimentConfig`.
5. Run the notebook cells. Dataset downloads are placed under `data/`; required original/base models are placed under `checkpoints/`; generated outputs are created under `results/`.

The exact commands and notebook filenames should be taken from the repository notebooks rather than inferred from this README.

## Limitations and interpretation notes

The supplied paper excerpt indicates the following limitations and caveats:

- Evidence is based on a single seed.
- Certain layers retain incomplete erasure traces.
- Calibration degrades.
- Baseline comparisons can be implementation-specific.
- The ImageWoof retraining reference is under-optimized.

These points should be considered when interpreting comparisons or reproducing the reported results.

## Citation

The paper is published, but its title, authors, venue, year, and URL were not supplied here. Please complete the citation details from `CITATION.cff` and the published paper before release:

```text
TODO: Insert the paper's canonical citation from CITATION.cff.
TODO: Confirm title, authors, venue, publication year, and URL.
```

For citation metadata, see [`CITATION.cff`](CITATION.cff).

## License

See [`License`](License).
