import numpy as np
# RGB in range [0,255]
# HSV: H in [0,360), S in [0,1], V in [0,1]

def rgb_pixel_to_hsv(r, g, b):
    r_, g_, b_ = r/255.0, g/255.0, b/255.0
    mx = max(r_, g_, b_)
    mn = min(r_, g_, b_)
    diff = mx - mn
    # Hue
    if diff == 0:
        h = 0.0
    elif mx == r_:
        h = (60 * ((g_ - b_) / diff) + 360) % 360
    elif mx == g_:
        h = (60 * ((b_ - r_) / diff) + 120) % 360
    else:
        h = (60 * ((r_ - g_) / diff) + 240) % 360
    # Saturation
    s = 0.0 if mx == 0 else diff / mx
    v = mx
    return h, s, v


def hsv_pixel_to_rgb(h, s, v):
    c = v * s
    x = c * (1 - abs(((h / 60.0) % 2) - 1))
    m = v - c
    if 0 <= h < 60:
        r1, g1, b1 = c, x, 0
    elif 60 <= h < 120:
        r1, g1, b1 = x, c, 0
    elif 120 <= h < 180:
        r1, g1, b1 = 0, c, x
    elif 180 <= h < 240:
        r1, g1, b1 = 0, x, c
    elif 240 <= h < 300:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x
    r, g, b = (r1 + m) * 255.0, (g1 + m) * 255.0, (b1 + m) * 255.0
    return r, g, b


def rgb_image_to_hsv(img_arr: np.ndarray) -> np.ndarray:
    R = img_arr[...,0].astype(np.float32)/255.0
    G = img_arr[...,1].astype(np.float32)/255.0
    B = img_arr[...,2].astype(np.float32)/255.0
    mx = np.maximum.reduce([R,G,B])
    mn = np.minimum.reduce([R,G,B])
    diff = mx - mn
    H = np.zeros_like(mx)
    S = np.zeros_like(mx)
    V = mx
    mask = diff != 0
    rmask = (mx==R) & mask
    gmask = (mx==G) & mask
    bmask = (mx==B) & mask
    H[rmask] = (60*((G[rmask]-B[rmask])/diff[rmask])+360)%360
    H[gmask] = (60*((B[gmask]-R[gmask])/diff[gmask])+120)%360
    H[bmask] = (60*((R[bmask]-G[bmask])/diff[bmask])+240)%360
    S[mx!=0] = diff[mx!=0]/mx[mx!=0]
    return np.stack([H,S,V],axis=-1)


def hsv_image_to_rgb(hsv: np.ndarray) -> np.ndarray:
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    C = V*S
    X = C*(1-np.abs(((H/60)%2)-1))
    m = V-C
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)
    # 6 sectors
    mask0 = (H>=0)&(H<60)
    mask1 = (H>=60)&(H<120)
    mask2 = (H>=120)&(H<180)
    mask3 = (H>=180)&(H<240)
    mask4 = (H>=240)&(H<300)
    mask5 = (H>=300)&(H<360)
    R[mask0], G[mask0], B[mask0] = C[mask0], X[mask0], 0
    R[mask1], G[mask1], B[mask1] = X[mask1], C[mask1], 0
    R[mask2], G[mask2], B[mask2] = 0, C[mask2], X[mask2]
    R[mask3], G[mask3], B[mask3] = 0, X[mask3], C[mask3]
    R[mask4], G[mask4], B[mask4] = X[mask4], 0, C[mask4]
    R[mask5], G[mask5], B[mask5] = C[mask5], 0, X[mask5]
    R=(R+m)*255.0
    G=(G+m)*255.0
    B=(B+m)*255.0
    return np.clip(np.stack([R,G,B],axis=-1),0,255).astype(np.uint8)


# YCbCr conversion (BT.601 full range approximation)
# We'll use Y' = 0.299 R + 0.587 G + 0.114 B

def rgb_to_ycbcr(img_arr: np.ndarray) -> np.ndarray:
    arr = img_arr.astype(np.float32)
    R = arr[..., 0]
    G = arr[..., 1]
    B = arr[..., 2]
    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 + (-0.168736) * R + (-0.331264) * G + 0.5 * B
    Cr = 128 + 0.5 * R + (-0.418688) * G + (-0.081312) * B
    out = np.stack([Y, Cb, Cr], axis=-1)
    return out


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    Y = ycbcr[..., 0]
    Cb = ycbcr[..., 1] - 128
    Cr = ycbcr[..., 2] - 128
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def rgb_to_gray(img_arr: np.ndarray) -> np.ndarray:
    arr = img_arr.astype(np.float32)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return np.clip(gray, 0, 255).astype(np.uint8)

def rgb_to_cmyk(img_arr: np.ndarray) -> np.ndarray:
    arr = img_arr.astype(np.float32) / 255.0
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    K = 1 - np.maximum.reduce([R, G, B])
    C = np.where(K < 1, (1 - R - K) / (1 - K + 1e-8), 0)
    M = np.where(K < 1, (1 - G - K) / (1 - K + 1e-8), 0)
    Y = np.where(K < 1, (1 - B - K) / (1 - K + 1e-8), 0)
    # scale về 0..255 cho dễ hiển thị
    C = (C * 255).astype(np.uint8)
    M = (M * 255).astype(np.uint8)
    Y = (Y * 255).astype(np.uint8)
    K = (K * 255).astype(np.uint8)
    return np.stack([C, M, Y, K], axis=-1)


def cmyk_to_rgb(cmyk: np.ndarray) -> np.ndarray:
    C = cmyk[..., 0].astype(np.float32) / 255.0
    M = cmyk[..., 1].astype(np.float32) / 255.0
    Y = cmyk[..., 2].astype(np.float32) / 255.0
    K = cmyk[..., 3].astype(np.float32) / 255.0
    R = (1 - np.minimum(1, C * (1 - K) + K)) * 255.0
    G = (1 - np.minimum(1, M * (1 - K) + K)) * 255.0
    B = (1 - np.minimum(1, Y * (1 - K) + K)) * 255.0
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)
