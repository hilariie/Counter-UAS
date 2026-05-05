from __future__ import annotations
import cv2
import numpy as np
import pygame

from cuas.viz.theme import Theme
from cuas.viz.ui_frame import UIFrame
from cuas.viz.geometry import bgr_to_surface, fit_letterbox
from .base import draw_panel_chrome


def draw_wide_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    ui_frame: UIFrame,
    theme: Theme,
) -> None:
    draw_panel_chrome(surface, rect, "", theme)

    ann = _annotate_wide_bgr(
        ui_frame.wide_frame_bgr,
        ui_frame.last_dets,
        ui_frame.last_rois,
        ui_frame.chosen_id,
    )
    frame_surf = bgr_to_surface(ann)
    scaled, dest = fit_letterbox(frame_surf, rect)
    surface.blit(scaled, dest)

    # HUD strip
    hud = (
        f"WIDE  fps={ui_frame.fps:.1f}  "
        f"f={ui_frame.frame_id}  "
        f"mode={ui_frame.mode}"
    )
    hud_surf = theme.font_body.render(hud, True, theme.accent_cyan)
    surface.blit(hud_surf, (rect.x + 6, rect.y + 4))


def _annotate_wide_bgr(
    frame: np.ndarray,
    dets: list,
    rois: list,
    chosen_id,
) -> np.ndarray:
    out = frame.copy()
    for (x1, y1, x2, y2) in rois:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
    for d in dets:
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (255, 255, 0), 2)
    if chosen_id is not None:
        cx = out.shape[1] // 2
        cy = out.shape[0] // 2
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    return out
