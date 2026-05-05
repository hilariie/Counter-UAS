from __future__ import annotations
import cv2
import numpy as np
import pygame

from cuas.viz.theme import Theme
from cuas.viz.ui_frame import UIFrame
from cuas.viz.geometry import bgr_to_surface, fit_letterbox
from .base import draw_panel_chrome

_STATE_COLORS = {
    "IDLE":      None,   # text_dim
    "WARMUP":    None,
    "ACQUIRING": "accent_amber",
    "LOCKED":    "ekf_green",
    "COMMIT":    "ekf_green",
}


def draw_narrow_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    ui_frame: UIFrame,
    theme: Theme,
) -> None:
    draw_panel_chrome(surface, rect, "", theme)

    cs = ui_frame.ctrl_state
    ann = _annotate_narrow_bgr(ui_frame.narrow_frame_bgr, cs)
    frame_surf = bgr_to_surface(ann)
    scaled, dest = fit_letterbox(frame_surf, rect)
    surface.blit(scaled, dest)

    # State badge pill upper-left
    state_name = cs.state_name
    color_key = _STATE_COLORS.get(state_name)
    if cs.lost:
        pill_color = theme.threat_red
        badge_text = f"[LOST]  rem={cs.state_remaining_s:.1f}s"
    elif color_key:
        pill_color = getattr(theme, color_key.lower())
        badge_text = f"{state_name}  rem={cs.state_remaining_s:.1f}s"
    else:
        pill_color = theme.text_dim
        badge_text = f"{state_name}  rem={cs.state_remaining_s:.1f}s"

    _draw_pill(surface, badge_text, pill_color, theme, rect.x + 6, rect.y + 6)

    # Intercept overlay — big banner at bottom of panel for demo clarity
    sol = ui_frame.intercept_solution
    is_commit = (cs.state_name == "COMMIT" and not cs.lost)
    if is_commit and sol is not None and sol.feasible and sol.time_to_intercept_s is not None:
        _draw_intercept_banner(surface, rect, sol.time_to_intercept_s, theme)


def _draw_intercept_banner(
    surface: pygame.Surface,
    rect: pygame.Rect,
    tti_s: float,
    theme: Theme,
) -> None:
    banner_h = 38
    banner_y = rect.bottom - banner_h - 2
    banner = pygame.Surface((rect.width - 4, banner_h), pygame.SRCALPHA)
    banner.fill((20, 180, 80, 200))   # green semi-transparent

    title = theme.font_h2.render("INTERCEPT SOLUTION COMPUTED", True, (8, 8, 8))
    tti   = theme.font_h2.render(f"TTI  {tti_s:.1f} s", True, (8, 8, 8))

    banner.blit(title, (8, 3))
    banner.blit(tti,   (8, 3 + title.get_height() + 2))

    surface.blit(banner, (rect.x + 2, banner_y))


def _draw_pill(
    surface: pygame.Surface,
    text: str,
    color: tuple,
    theme: Theme,
    x: int,
    y: int,
) -> None:
    label = theme.font_mono.render(text, True, (8, 8, 8))
    pw = label.get_width() + 12
    ph = label.get_height() + 6
    pill = pygame.Surface((pw, ph), pygame.SRCALPHA)
    pill.fill((*color, 220))
    pill.blit(label, (6, 3))
    surface.blit(pill, (x, y))


def _annotate_narrow_bgr(frame: np.ndarray, ctrl_state) -> np.ndarray:
    out = frame.copy()
    det = ctrl_state.det
    if det is not None:
        x, y, w, h = det.bbox_xywh
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.drawMarker(out, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 15, 1)
        tag = f"id={det.track_id} c={det.confidence:.2f}"
        cv2.putText(out, tag, (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1)
    ih, iw = out.shape[:2]
    cv2.drawMarker(out, (iw // 2, ih // 2), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
    if ctrl_state.lost:
        cv2.putText(out, "[LOST]", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out
