from __future__ import annotations
import math
from math import degrees
from typing import Tuple

import numpy as np
import pygame


def bgr_to_surface(bgr: np.ndarray) -> pygame.Surface:
    rgb = bgr[:, :, ::-1]
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


def fit_letterbox(
    surface: pygame.Surface,
    rect: pygame.Rect,
) -> Tuple[pygame.Surface, pygame.Rect]:
    sw, sh = surface.get_size()
    rw, rh = rect.width, rect.height
    scale = min(rw / sw, rh / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    if nw != sw or nh != sh:
        scaled = pygame.transform.smoothscale(surface, (nw, nh))
    else:
        scaled = surface
    ox = rect.x + (rw - nw) // 2
    oy = rect.y + (rh - nh) // 2
    return scaled, pygame.Rect(ox, oy, nw, nh)


_MINIMAP_RANGE_M = 300.0


def ned_to_minimap_px(
    ned: np.ndarray,
    origin_ned: np.ndarray,
    rect: pygame.Rect,
    range_m: float = _MINIMAP_RANGE_M,
) -> Tuple[int, int]:
    """NED world → minimap pixel. +N is up, +E is right."""
    rel = ned[:2] - origin_ned[:2]
    cx = rect.x + rect.width // 2
    cy = rect.y + rect.height // 2
    scale = min(rect.width, rect.height) / (2.0 * range_m)
    px = int(cx + rel[1] * scale)   # E → right
    py = int(cy - rel[0] * scale)   # N → up
    return px, py


def minimap_px_to_ne(
    px: int,
    py: int,
    origin_ned: np.ndarray,
    rect: pygame.Rect,
    range_m: float = _MINIMAP_RANGE_M,
) -> Tuple[float, float]:
    cx = rect.x + rect.width // 2
    cy = rect.y + rect.height // 2
    scale = min(rect.width, rect.height) / (2.0 * range_m)
    n = -(py - cy) / scale + origin_ned[0]
    e = (px - cx) / scale + origin_ned[1]
    return n, e


def cov2d_to_ellipse_params(
    P: np.ndarray,
    n_sigma: float = 2.0,
) -> Tuple[float, float, float]:
    """Return (semi_major_px, semi_minor_px, theta_rad) for a 2x2 covariance.

    Uses eigh (symmetric eigenvalue decomp). Returns (0, 0, 0) for non-PSD input.
    The caller is responsible for applying a per-pixel scale factor.
    """
    try:
        vals, vecs = np.linalg.eigh(P)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0

    if np.any(vals < 0):
        return 0.0, 0.0, 0.0

    # eigh returns ascending order — largest eigenvalue is vecs[:,1]
    major = float(np.sqrt(vals[1])) * n_sigma
    minor = float(np.sqrt(vals[0])) * n_sigma
    theta = float(np.arctan2(vecs[1, 1], vecs[0, 1]))
    # Eigenvectors have no preferred sign; normalise to [-pi/2, pi/2] so the
    # caller gets a consistent angle regardless of numpy's sign choice.
    if theta < -math.pi / 2:
        theta += math.pi
    elif theta > math.pi / 2:
        theta -= math.pi
    return major, minor, theta
