# CMMD Metric
An optimised implementation of CLIP Maximum Mean Discrepancy (CMMD) for evaluating image generation models, proposed in [Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/abs/2401.09603). 
According to the paper, CMMD is a better metric compared to FID and as it tries to mitigate some longstanding issues of FID.

This implementation has been adapted from this [codebase](https://github.com/sayakpaul/cmmd-pytorch).

## Installation

Do the following steps to install this package"

1. Install this package using pip:
   ```bash
   pip install git+https://github.com/pfgurus/cmmd
   ```

   Or alternatively 
   clone this repository and install it using pip:
   ```bash
    cd path/to/this/repo
    pip install -e .
    ```

## Usage
See examples/cmmd_test.py

