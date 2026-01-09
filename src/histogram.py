import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def get_hist_image(arr: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(3,2), dpi=100)
    ax.clear()
    if arr.ndim == 3 and arr.shape[2] == 3:
        for i, col in enumerate(('r','g','b')):
            hist, bins = np.histogram(arr[...,i].flatten(), 256, (0,255))
            ax.plot(bins[:-1], hist, color=col)
    else:
        hist, bins = np.histogram(arr.flatten(), 256, (0,255))
        ax.plot(bins[:-1], hist, color='black')
    ax.set_xlim(0,255)
    ax.set_yticks([])
    ax.set_xticks([0,128,255])
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
