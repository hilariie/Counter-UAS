from __future__ import annotations
from typing import Optional

import cv2
import numpy as np
import pygame

from cuas.viz.theme import Theme, THEME_DEFAULT
from cuas.viz.ui_frame import UIFrame
from cuas.viz.alerts import RadarFailAlert
from cuas.viz.panels import (
    draw_wide_panel,
    draw_narrow_panel,
    draw_threat_list_panel,
    draw_minimap_panel,
)
from cuas.viz.panels.base import draw_panel_chrome


class OperatorUI:
    def __init__(
        self,
        size: tuple[int, int] = (1280, 720),
        title: str = "Counter-UAS Operator UI",
        theme: Theme = THEME_DEFAULT,
        save_video: Optional[str] = None,
        video_fps: float = 20.0,
    ):
        self._size   = size
        self._title  = title
        self._theme  = theme
        self._screen: Optional[pygame.Surface] = None
        self._radar_alert: Optional[RadarFailAlert] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._save_video = save_video
        self._video_fps  = video_fps

    def init(self) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode(self._size)
        pygame.display.set_caption(self._title)
        self._theme.init()
        self._radar_alert = RadarFailAlert(screen_size=self._size)
        if self._save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self._save_video, fourcc, self._video_fps, self._size
            )
            print(f"[video] saving to {self._save_video} @ {self._video_fps:.0f} fps")

    def pump_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def render(self, ui_frame: UIFrame) -> None:
        screen = self._screen
        theme  = self._theme
        layout = theme.layout

        screen.fill(theme.bg)

        draw_wide_panel(screen, layout["WIDE"], ui_frame, theme)
        draw_narrow_panel(screen, layout["NARROW"], ui_frame, theme)
        draw_threat_list_panel(screen, layout["THREAT_LIST"], ui_frame, theme)
        draw_minimap_panel(screen, layout["MINIMAP"], ui_frame, theme)
        _draw_status_strip(screen, layout["STATUS_STRIP"], ui_frame, theme)

        self._radar_alert.update(ui_frame.radar_alive, ui_frame.t_elapsed)
        self._radar_alert.draw(screen)

        pygame.display.flip()

        if self._writer is not None:
            # surfarray returns (W, H, 3) RGB; VideoWriter needs (H, W, 3) BGR
            rgb = pygame.surfarray.array3d(screen)
            bgr = rgb.swapaxes(0, 1)[:, :, ::-1]
            self._writer.write(bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"[video] saved → {self._save_video}")
        pygame.quit()


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
        f"tracks:  {len(ui_frame.ranked)}",
        f"chosen:  #{ui_frame.chosen_id}",
    ]
    y = rect.y + 20
    for line in lines:
        surf = theme.font_mono.render(line, True, theme.text_primary)
        surface.blit(surf, (rect.x + 8, y))
        y += surf.get_height() + 4
