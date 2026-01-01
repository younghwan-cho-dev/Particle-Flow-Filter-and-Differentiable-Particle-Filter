# PF-PF: Particle Flow Particle Filter

Implementation of Particle Flow Particle Filter methods for JP Morgan MLCOE interview rounds:

> Kalman Filter
> Extended Kalman Filter
> Unscented Kalman Filter
> Bootstrap Particle Filter
> Li, Yunpeng, and Mark Coates (2017). "Particle filtering with invertible particle flow." 

This repository contains the code for the JP Morgan MLCOE TSRL 2026 internship question 2 written by Younghwan Cho.

## Setup

### Create conda environment
```bash
conda env create -f environment.yml
conda activate pf_pf
```

## Project Structure
```
pf_pf/
├── src/                          # TensorFlow implementations
│   ├── __init__.py
│   ├── prob_ssm.py               # ProbSSM base class
│   ├── utils.py                  # Utilities (resampling, metrics, plotting)
│   │
│   ├── models/
│   │   └── __init__.py           # SSM model definitions (TensorFlow)
│   │
│   ├── filters/
│   │   └── __init__.py           # include KF, EKF, UKF, PF (TensorFlow) (will be fully migrated to Numpy implementation)
│   │
│   └── flows/
│       └── __init__.py           # EDH flow equations (TensorFlow)
│
├── src_numpy/                    # NumPy implementations
│   ├── __init__.py
│   ├── numpy_ssm.py              # NumpySSM base class
│   ├── numpy_models.py           # SSM model definitions
│   ├── numpy_pf.py               # Bootstrap particle filter
│   ├── numpy_edh_flow.py         # EDH flow equations
│   └── numpy_pf_edh.py           # EDH PF-PF
│
├── notebooks/
│   ├── 01_test_lgssm.ipynb       # Test LGSSM and KF
│   ├── 02_test_nonlinear.ipynb   # Test nonlinear models
│   ├── 03_test_edh_pfpf_paper.ipynb  # Replicate paper results
│   ├── 03_test_edh_pfpf.ipynb    # EDH PF-PF experiments
│   └── run_test_pf_pf_numpy.ipynb
│
├── test_filters.py               # Unit tests (pytest)
├── environment.yml
└── README.md
```

## Usage

### Quick start
```python

import sys
sys.path.append('..')

import numpy as np
import tensorflow as tf


from src_numpy import (
    particle_filter_numpy,
    make_random_walk_ssm_np,
)

ssm = make_random_walk_ssm_np(Q=0.2, R=0.5)
xs, ys = ssm.simulate(T=200, seed=42)
ms, Ps, info = particle_filter_numpy(ssm, ys, num_particles=1000, seed=123)
```

## Testing
```bash
pytest test_filters.py -v
```

Tests include:
- KF covariance convergence and validity
- EKF/UKF equivalence to KF under linear Gaussian SSM
- PF approximation to KF with systematic/multinomial resampling
- Weight validity and reproducibility

## Implemented Algorithms

| Algorithm | TensorFlow (`src/`) | NumPy (`src_numpy/`) |
|-----------|---------------------|----------------------|
| Kalman Filter | ✓ | |
| Extended Kalman Filter | ✓ | |
| Unscented Kalman Filter | ✓ | |
| Bootstrap Particle Filter | ✓ | ✓ |
| EDH Flow | | ✓ |
| EDH PF-PF | | ✓ |
