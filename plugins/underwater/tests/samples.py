"""Synthetic data for the tests: underwater images made with the image formation model, image
sequences moving over a texture, depth logs, aligned cameras at known depths."""
from __future__ import annotations

import math

import cv2
import numpy as np

# the water of the tests, per channel B, G, R (OpenCV order): red absorbed fast, blue-green backscatter
BETA_D = np.array([0.10, 0.07, 0.45])
BETA_B = np.array([0.40, 0.35, 0.25])
B_INF = np.array([0.30, 0.22, 0.02])


def texture(h: int, w: int, seed: int = 0, cells: tuple[int, int] = (20, 26)) -> np.ndarray:
    """Reflectance (linear, BGR): patches of random colours with a fine texture."""
    rs = np.random.default_rng(seed)
    base = rs.uniform(0.08, 0.75, (cells[0], cells[1], 3)).astype(np.float32)
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_NEAREST)
    fine = cv2.GaussianBlur(rs.standard_normal((h, w)).astype(np.float32), (0, 0), 1.2)
    fine = 1.0 + 0.6 * fine / (fine.std() + 1e-6) * 0.25
    j = np.clip(base * fine[..., None], 0.02, 0.95)
    # crevices and shadows: the dark pixels a scene under water always has somewhere
    holes = cv2.dilate((rs.uniform(size=(h, w)) < 0.004).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    j[holes] = 0.004
    return j.astype(np.float32)


def ranges(h: int, w: int, near: float, far: float, seed: int = 0) -> np.ndarray:
    """A sloping bottom seen obliquely: far at the top of the image, near at the bottom, a mound."""
    rs = np.random.default_rng(seed)
    z = np.linspace(far, near, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = rs.uniform(0.3, 0.7) * w, rs.uniform(0.4, 0.8) * h
    z -= 0.25 * (far - near) * np.exp(-(((xx - cx) / (0.2 * w)) ** 2 + ((yy - cy) / (0.2 * h)) ** 2)).astype(np.float32)
    return np.maximum(z, 0.3 * near)


def underwater(j: np.ndarray, z: np.ndarray, exposure: float = 1.0, noise: float = 0.003, seed: int = 0) -> np.ndarray:
    """The linear image a camera sees under the water of the tests."""
    rs = np.random.default_rng(seed)
    zz = z[..., None]
    img = j * np.exp(-BETA_D * zz) + B_INF * (1.0 - np.exp(-BETA_B * zz))
    img = exposure * img + noise * rs.standard_normal(img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def srgb(lin: np.ndarray) -> np.ndarray:
    from ppg_underwater.colour import to_srgb

    return to_srgb(lin)


def blobs(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A grey texture with features at several scales (stones on sand)."""
    rs = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for sigma, amp in ((1.0, 0.5), (3.0, 1.0), (9.0, 1.5)):
        n = cv2.GaussianBlur(rs.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
        img += amp * n / (n.std() + 1e-6)
    img = (img - img.min()) / (img.max() - img.min())
    return (40 + 180 * img).astype(np.uint8)


def random_rotation(seed: int = 0) -> np.ndarray:
    rs = np.random.default_rng(seed)
    q = rs.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def looking_down(heading: float = 0.0) -> np.ndarray:
    """world_from_cam rotation of a camera looking straight down (OpenCV axes: z forward)."""
    c, s = math.cos(heading), math.sin(heading)
    # camera x along the heading, camera z down, camera y completing the right-handed frame
    x = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
