# CMMD Metric
An optimised implementation of CLIP Maximum Mean Discrepancy (CMMD) for evaluating image generation models, proposed in [Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/abs/2401.09603). 
According to the paper, CMMD is a better metric compared to FID and as it tries to mitigate some longstanding issues of FID.

This implementation has been adapted from this [codebase](https://github.com/sayakpaul/cmmd-pytorch).

## Installation

Do the following steps to install this package

1. Make sure you have installed the following packages:
   - torch
   - torchvision
   
   Use [official PyTorch installation guide](https://pytorch.org/get-started/locally/) to install PyTorch.

2. Install this package using pip:
   ```bash
   pip install git+https://github.com/pfgurus/cmmd
   ```

   Or alternatively 
   clone this repository and install it using pip:
   ```bash
    cd path/to/this/repo
    pip install .
    ```

## Usage
```python
import cmmd

cmmd_metric = cmmd.CMMD(device=device)

# compute real and fake embeddings given batch of real and fake image tensors
cmmd_metric.update(real_image, kind='real')
cmmd_metric.update(genreated_image, kind='fake')

cmmd_value = cmmd_metric.compute().item()
```

See examples/cmmd_test.py for more.

## Changes from original
- Uniform batch compute to ensure optimal performance
- Use of FP16 version for the ViT model to compute embedddings
- Use of SDPA implementation for attention module.

