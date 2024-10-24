import os

import torch
import numpy as np
from PIL import Image

from cmmd import CMMD


def test_cmmd():
    test_data = os.path.join(os.path.dirname(__file__), '..', 'assets/cmmd')
    real_dir = os.path.join(test_data, "reference_images")
    generated_dir = os.path.join(test_data, "generated_images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cmmd_metric = CMMD(device=device, range_converter=range_255_1)

    real_imgs = [Image.open(os.path.join(real_dir, f)) for f in os.listdir(real_dir)]
    real_img_t = torch.from_numpy(np.asarray(real_imgs)).permute(0, 3, 1, 2).float()

    generated_img = [Image.open(os.path.join(generated_dir, f)) for f in os.listdir(generated_dir)]
    generated_img_t = torch.from_numpy(np.array(generated_img)).permute(0, 3, 1, 2).float()

    cmmd_metric.update(real_img_t, kind="real")
    cmmd_metric.update(generated_img_t, kind="fake")
    cmmd = cmmd_metric.compute().item()

    # See reference value here: https://github.com/sayakpaul/cmmd-pytorch
    assert np.allclose(cmmd, 7.696, atol=0.1)


def range_255_1(x):
    """
    Convert from input range [0, 255] to [0, 1].
    """
    return x * (1 / 255)
