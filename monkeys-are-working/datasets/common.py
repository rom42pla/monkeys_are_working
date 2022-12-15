from os.path import exists

import cv2
import einops
import numpy as np


def read_image(path: str, precision: int = 32) -> np.ndarray:
    assert exists(path), f"{path} does not exists"
    assert precision in {16, 32}, f"precision must be either 16 or 32 bits"
    dtype = np.float16 if precision == 16 else np.float32
    image = cv2.cvtColor(cv2.imread(path),
                         cv2.COLOR_BGR2RGB)
    image = einops.rearrange(image, "h w c -> c h w")
    image = image / 255
    image = image.astype(dtype)
    return image
