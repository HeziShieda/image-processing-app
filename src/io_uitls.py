import numpy as np
from PIL import Image

def pil_to_np(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    return arr.astype(np.uint8)

def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)
