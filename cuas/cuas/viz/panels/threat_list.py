from __future__ import annotations
import math

import numpy as np
import pygame

from cuas.viz.theme import Theme
from cuas.viz.ui_frame import UIFrame
from cuas.state.types import SensorMask
from .base import draw_panel_chrome

_NIS_GREEN  = 5.99
_NIS_AMBER  = 11.34
_MAX_ROWS   = 5   # 5 rows; each gets _ROW_H px so TTI second-line fits
_ROW_H      = 36

# Fixed pixel offsets from rect.x for each column
_COL_RK   = 6
_COL_ID   = 28
_COL_RNG  = 72
_COL_BRG  = 122
_COL_SCR  = 172


def draw_threat_list_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    ui_frame: UIFrame,
    theme: Theme,
) -> None:
    draw_panel_chrome(surface, rect, "THREAT LIST", theme)

    # Right-anchored columns (relative to rect.x)
    col_chip = rect.x + rect.width - 70   # start of R chip
    col_dot  = rect.x + rect.width - 10   # centre of NIS dot

    # Header
    for txt, rx in (("RK", _COL_RK), ("ID", _COL_ID), ("RNG", _COL_RNG),
                    ("BRG", _COL_BRG), ("SCR", _COL_SCR)):
        surface.blit(theme.font_mono.render(txt, True, theme.text_dim),
                     (rect.x + rx, rect.y + 20))
    for i, ch in enumerate(("R", "B", "S")):
        surface.blit(theme.font_mono.render(ch, True, theme.text_dim),
                     (col_chip + i * 20, rect.y + 20))

    est_by_id = {se.track_id: se for se in ui_frame.state_estimates}
    y0 = rect.y + 38

    for rank, rt in enumerate(ui_frame.ranked[:_MAX_ROWS], start=1):
        t = rt.track
        row_y = y0 + (rank - 1) * _ROW_H
        if row_y + _ROW_H > rect.bottom - 4:
            break

        is_chosen = (t.id == ui_frame.chosen_id)
        is_commit = (ui_frame.ctrl_state.state_name == "COMMIT" and
                     ui_frame.ctrl_state.target_id == t.id)

        if is_chosen:
            tint = pygame.Surface((rect.width - 4, _ROW_H), pygame.SRCALPHA)
            tint.fill((244, 212, 52, 40))
            surface.blit(tint, (rect.x + 2, row_y))
            pygame.draw.rect(surface, theme.chosen_yellow,
                             (rect.x + 2, row_y, 3, _ROW_H))

        se = est_by_id.get(t.id)

        if se is not None:
            rng = f"{np.linalg.norm(se.position_ned):.0f}m"
        elif t.range_m is not None:
            rng = f"{t.range_m:.0f}m"
        else:
            rng = "---"

        text_color = theme.chosen_yellow if is_chosen else theme.text_primary
        text_y = row_y + 4

        def _col(txt, offset, color=None):
            surface.blit(theme.font_mono.render(txt, True, color or text_color),
                         (rect.x + offset, text_y))

        _col(f"{rank}", _COL_RK)
        _col(f"#{t.id}", _COL_ID)
        _col(rng, _COL_RNG)
        _col(f"{math.degrees(t.az_rad):+.1f}°", _COL_BRG)
        _col(f"{rt.score:.2f}", _COL_SCR)

        for i, (chip_text, chip_col) in enumerate(_sensor_chips(se, theme)):
            surface.blit(theme.font_mono.render(chip_text, True, chip_col),
                         (col_chip + i * 20, text_y))

        diag = ui_frame.filter_diags.get(t.id)
        pygame.draw.circle(surface, _nis_color(diag, theme), (col_dot, row_y + _ROW_H // 2), 5)

        # TTI second line — lower half of same row, only for COMMIT target
        if is_commit and ui_frame.intercept_solution is not None:
            sol = ui_frame.intercept_solution
            if sol.feasible and sol.time_to_intercept_s is not None:
                tti_text = f"  TTI={sol.time_to_intercept_s:.0f}s  [INTERCEPT SOLUTION]"
                tti_color = theme.accent_amber
            else:
                reason = sol.reason.value[:10] if sol.reason else "INFEASIBLE"
                tti_text = f"  [{reason}]"
                tti_color = theme.threat_red
            tti_surf = theme.font_mono.render(tti_text, True, tti_color)
            surface.blit(tti_surf, (rect.x + _COL_RK,
                                    row_y + _ROW_H - tti_surf.get_height() - 3))


def _sensor_chips(se, theme: Theme) -> list:
    if se is None:
        return [("R", theme.text_dim), ("B", theme.text_dim), ("S", theme.text_dim)]
    return [
        (letter, theme.ekf_green if flag in se.sensors_used else theme.text_dim)
        for flag, letter in ((SensorMask.RADAR, "R"), (SensorMask.BEARING, "B"), (SensorMask.SFM, "S"))
    ]


def _nis_color(diag, theme: Theme) -> tuple:
    if diag is None:
        return theme.text_dim
    _, nis, _ = diag
    if nis <= _NIS_GREEN:
        return theme.ekf_green
    if nis <= _NIS_AMBER:
        return theme.accent_amber
    return theme.threat_red
