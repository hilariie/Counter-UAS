from __future__ import annotations
import math
from math import degrees

import numpy as np
import pygame

from cuas.viz.theme import Theme
from cuas.viz.ui_frame import UIFrame
from cuas.viz.geometry import ned_to_minimap_px, cov2d_to_ellipse_params
from .base import draw_panel_chrome

_RANGE_M = 300.0
_RINGS_M = (100.0, 200.0)


def draw_minimap_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    ui_frame: UIFrame,
    theme: Theme,
) -> None:
    draw_panel_chrome(surface, rect, "MINIMAP  (NE top-down, 300m half-extent)", theme)

    cx = rect.x + rect.width // 2
    cy = rect.y + rect.height // 2
    scale = min(rect.width, rect.height) / (2.0 * _RANGE_M)

    # 1. Range rings + labels + N tick
    for r_m in _RINGS_M:
        r_px = int(r_m * scale)
        pygame.draw.circle(surface, theme.panel_border, (cx, cy), r_px, 1)
        lbl = theme.font_mono.render(f"{r_m:.0f}m", True, theme.text_dim)
        surface.blit(lbl, (cx + r_px + 2, cy - lbl.get_height() // 2))
    # N tick
    n_px = int(_RINGS_M[-1] * scale)
    pygame.draw.line(surface, theme.text_dim, (cx, cy - n_px - 8), (cx, cy - n_px - 20), 1)
    n_label = theme.font_mono.render("N", True, theme.text_dim)
    surface.blit(n_label, (cx - n_label.get_width() // 2, cy - n_px - 32))

    origin = ui_frame.ownship_ned

    # 2. Ownship triangle
    _draw_ownship(surface, cx, cy, ui_frame.ownship_yaw_rad, theme)

    est_by_id = {se.track_id: se for se in ui_frame.state_estimates}

    # Score percentiles for color
    scores = [rt.score for rt in ui_frame.ranked] if ui_frame.ranked else [0.0]
    s_min, s_max = min(scores), max(scores)

    for rt in ui_frame.ranked:
        t  = rt.track
        se = est_by_id.get(t.id)

        # Color by score percentile
        frac = (rt.score - s_min) / max(s_max - s_min, 1e-6)
        if frac >= 0.66:
            dot_color = theme.threat_red
        elif frac >= 0.33:
            dot_color = theme.accent_amber
        else:
            dot_color = theme.text_dim

        if se is not None:
            # EKF position in world NED → correct absolute range + bearing
            px, py = ned_to_minimap_px(se.position_ned, origin, rect, _RANGE_M)
        elif t.range_m is not None:
            # Fallback: radar az + range → approximate NED relative to ownship
            yaw = ui_frame.ownship_yaw_rad
            az  = t.az_rad  # relative to ownship nose
            r   = t.range_m
            # az_world = ownship_yaw + az_body; NED: N=cos, E=sin
            az_world = yaw + az
            rel_ned = np.array([r * math.cos(az_world), r * math.sin(az_world), 0.0])
            px, py = ned_to_minimap_px(origin + rel_ned, origin, rect, _RANGE_M)
        else:
            continue

        if not (rect.x <= px < rect.right and rect.y <= py < rect.bottom):
            continue

        pygame.draw.circle(surface, dot_color, (px, py), 4)

        # ID label next to dot
        id_lbl = theme.font_mono.render(f"#{t.id}", True, dot_color)
        surface.blit(id_lbl, (px + 6, py - id_lbl.get_height() // 2))

        # Velocity vector from EKF (not available from radar-only): 0.5*|v| clamped 30px
        if se is not None:
            v = se.velocity_ned[:2]
            v_len = np.linalg.norm(v)
            if v_len > 0.1:
                v_px = min(30, int(0.5 * v_len * scale))
                v_unit = v / v_len
                ex = int(px + v_unit[1] * v_px)
                ey = int(py - v_unit[0] * v_px)
                pygame.draw.line(surface, dot_color, (px, py), (ex, ey), 2)

    # 4. Chosen ring
    if ui_frame.chosen_id is not None:
        se = est_by_id.get(ui_frame.chosen_id)
        if se is not None:
            px, py = ned_to_minimap_px(se.position_ned, origin, rect, _RANGE_M)
            pygame.draw.circle(surface, theme.chosen_yellow, (px, py), 8, 2)

    # 5. COMMIT-only covariance ellipse
    is_commit = ui_frame.ctrl_state.state_name == "COMMIT"
    commit_id = ui_frame.ctrl_state.target_id if is_commit else None
    if commit_id is not None:
        se = est_by_id.get(commit_id)
        if se is not None:
            _draw_cov_ellipse(surface, se, origin, rect, scale, theme)

    # 6. Intercept arc
    sol = ui_frame.intercept_solution
    if is_commit and sol is not None and sol.feasible and sol.intercept_point_ned is not None:
        _draw_intercept_arc(surface, origin, sol.intercept_point_ned, rect, theme)


def _draw_ownship(
    surface: pygame.Surface,
    cx: int,
    cy: int,
    yaw_rad: float,
    theme: Theme,
) -> None:
    size = 10
    # NED body frame: tip at +N [size,0], base corners at [-size/2, ±size/2].
    # Screen mapping: px = cx + E, py = cy - N → tip (size,0) → (cx, cy-size) = top ✓
    # yaw=180° rotates tip to (-size,0) → (cx, cy+size) = South ✓
    pts_body = np.array([[size, 0], [-size // 2, -size // 2], [-size // 2, size // 2]], dtype=float)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    rotated = (rot @ pts_body.T).T
    pts = [(int(cx + p[1]), int(cy - p[0])) for p in rotated]
    pygame.draw.polygon(surface, theme.accent_cyan, pts)


def _draw_cov_ellipse(
    surface: pygame.Surface,
    se,
    origin: np.ndarray,
    rect: pygame.Rect,
    scale: float,
    theme: Theme,
) -> None:
    P2 = se.covariance[:2, :2]
    major_m, minor_m, theta_rad = cov2d_to_ellipse_params(P2, n_sigma=2.0)
    if major_m == 0.0:
        return

    major_px = max(2, int(major_m * scale))
    minor_px = max(1, int(minor_m * scale))

    scratch_w = max(4, major_px * 2 + 4)
    scratch_h = max(4, minor_px * 2 + 4)
    scratch = pygame.Surface((scratch_w, scratch_h), pygame.SRCALPHA)
    pygame.draw.ellipse(scratch, (232, 58, 72, 60),
                        (0, scratch_h // 2 - minor_px, scratch_w, minor_px * 2))
    pygame.draw.ellipse(scratch, (232, 58, 72, 200),
                        (0, scratch_h // 2 - minor_px, scratch_w, minor_px * 2), 1)

    rotated = pygame.transform.rotate(scratch, -degrees(theta_rad))
    px, py = ned_to_minimap_px(se.position_ned, origin, rect, _RANGE_M)
    dest = rotated.get_rect(center=(px, py))
    surface.blit(rotated, dest)


def _draw_intercept_arc(
    surface: pygame.Surface,
    origin: np.ndarray,
    intercept_ned: np.ndarray,
    rect: pygame.Rect,
    theme: Theme,
) -> None:
    ox = rect.x + rect.width // 2
    oy = rect.y + rect.height // 2
    scale = min(rect.width, rect.height) / (2.0 * _RANGE_M)

    ix, iy = ned_to_minimap_px(intercept_ned, origin, rect, _RANGE_M)

    # Dashed line: 8 on / 4 off
    dx, dy = ix - ox, iy - oy
    length = math.hypot(dx, dy)
    if length < 2:
        return
    ux, uy = dx / length, dy / length
    dash_on, dash_off = 8, 4
    step = dash_on + dash_off
    t = 0.0
    while t < length:
        t0 = t
        t1 = min(t + dash_on, length)
        x0 = int(ox + ux * t0)
        y0 = int(oy + uy * t0)
        x1 = int(ox + ux * t1)
        y1 = int(oy + uy * t1)
        pygame.draw.line(surface, theme.accent_amber, (x0, y0), (x1, y1), 2)
        t += step

    # Diamond at endpoint
    d = 6
    diamond = [(ix, iy - d), (ix + d, iy), (ix, iy + d), (ix - d, iy)]
    pygame.draw.polygon(surface, theme.accent_amber, diamond)


def _draw_status_strip(
    surface: pygame.Surface,
    rect: pygame.Rect,
    ui_frame: UIFrame,
    theme: Theme,
) -> None:
    draw_panel_chrome(surface, rect, "STATUS", theme)
    lines = [
        f"mode:    {ui_frame.mode}",
        f"fps:     {ui_frame.fps:.1f}",
        f"elapsed: {ui_frame.t_elapsed:.1f}s",
        f"lost:    {ui_frame.n_lost}",
    ]
    y = rect.y + 20
    for line in lines:
        surf = theme.font_mono.render(line, True, theme.text_primary)
        surface.blit(surf, (rect.x + 8, y))
        y += surf.get_height() + 4
