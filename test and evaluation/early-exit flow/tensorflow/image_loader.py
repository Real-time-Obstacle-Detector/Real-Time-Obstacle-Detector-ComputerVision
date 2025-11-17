import os
import glob
import numpy as np
from PIL import Image

def test_images(folder_path, pattern="*.jpg", target_size=(640, 640), normalize=True):
    """
    Yields images as np.float32 with shape (1, 640, 640, 3) from the given folder.

    Args:
        folder_path (str): Directory containing .jpg files.
        pattern (str): Glob pattern for file names (default: '*.jpg').
        target_size (tuple): (width, height) to resize to (default: (640, 640)).
        normalize (bool): If True, divide by 255.0.

    Yields:
        np.ndarray: Array with shape (1, 640, 640, 3), dtype=float32.
    """
    paths = sorted(glob.glob(os.path.join(folder_path, pattern)))
    for p in paths:
        img = Image.open(p).convert("RGB").resize(target_size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if normalize:
            arr = arr / 255.0
        yield np.expand_dims(arr, axis=0)  # (1, 640, 640, 3)