import numpy as np
from color_space import rgb_to_ycbcr, ycbcr_to_rgb


def pad_image_channel(channel: np.ndarray, pad_h: int, pad_w: int, mode='edge') -> np.ndarray:
    return np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)


def convolve2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # kernel assumed 2D, kernel center at floor(k/2)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = pad_image_channel(channel, ph, pw)
    out = np.zeros_like(channel, dtype=np.float32)
    # flip kernel for convolution
    kflip = np.flipud(np.fliplr(kernel))
    h, w = channel.shape
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kflip)
    return out


def apply_convolution(img_arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Works for grayscale or color (H,W) or (H,W,3)
    if img_arr.ndim == 2:
        res = convolve2d(img_arr.astype(np.float32), kernel)
        return np.clip(res, 0, 255).astype(np.uint8)
    else:
        channels = []
        for c in range(3):
            conv = convolve2d(img_arr[..., c].astype(np.float32), kernel)
            channels.append(conv)
        out = np.stack(channels, axis=-1)
        return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-size//2 + 1., size//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def median_filter(img_arr: np.ndarray, ksize: int) -> np.ndarray:
    ph = pw = ksize // 2
    if img_arr.ndim == 2:
        padded = pad_image_channel(img_arr, ph, pw)
        h, w = img_arr.shape
        out = np.zeros_like(img_arr)
        for i in range(h):
            for j in range(w):
                region = padded[i:i+ksize, j:j+ksize]
                out[i, j] = np.median(region)
        return out
    else:
        h, w, _ = img_arr.shape
        out = np.zeros_like(img_arr)
        for c in range(3):
            padded = pad_image_channel(img_arr[..., c], ph, pw)
            for i in range(h):
                for j in range(w):
                    out[i, j, c] = np.median(padded[i:i+ksize, j:j+ksize])
        return out


def unsharp_mask(img_arr: np.ndarray, kernel_size=5, sigma=1.0, amount=1.0) -> np.ndarray:
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred = apply_convolution(img_arr, kernel)
    # Convert to float for arithmetic
    sharp = img_arr.astype(np.float32) + amount * (img_arr.astype(np.float32) - blurred.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)


def histogram_equalize_gray(gray: np.ndarray) -> np.ndarray:
    h = np.bincount(gray.flatten(), minlength=256)
    pdf = h / np.sum(h)
    cdf = np.cumsum(pdf)
    lut = np.floor(255 * cdf + 0.5).astype(np.uint8)
    out = lut[gray]
    return out


def histogram_equalize_color(img_arr: np.ndarray) -> np.ndarray:
    # Convert RGB->YCbCr, equalize Y, convert back
    ycbcr = rgb_to_ycbcr(img_arr)
    Y = ycbcr[..., 0].astype(np.uint8)
    Y_eq = histogram_equalize_gray(Y)
    ycbcr[..., 0] = Y_eq
    rgb = ycbcr_to_rgb(ycbcr)
    return rgb


def adjust_brightness_contrast(img_arr: np.ndarray, brightness=0.0, contrast=1.0) -> np.ndarray:
    # brightness in range [-255,255], contrast multiplier around 1.0
    out = img_arr.astype(np.float32) * contrast + brightness
    return np.clip(out, 0, 255).astype(np.uint8)
