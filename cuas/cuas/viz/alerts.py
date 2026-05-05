from __future__ import annotations
import math

import pygame


class RadarFailAlert:
    FADE_IN_S  = 0.35
    FADE_OUT_S = 0.60
    BLINK_HZ   = 1.6
    NOMINAL_PILL_S = 2.5

    def __init__(self, screen_size: tuple[int, int] = (1280, 720)):
        self._w, self._h = screen_size
        self._active    = False
        self._fade_t    = 0.0   # 0 → 1
        self._was_alive = True
        self._failure_t: float | None = None
        self._t_now     = 0.0
        self._last_t    = 0.0
        self._nominal_remaining = 0.0  # seconds left on green pill

        # Built once; blit each frame with varying alpha
        self._scanline_surf: pygame.Surface | None = None

        # Corner chevron cache: (surface, dest_rect) per corner index 0-3
        self._chevrons: list | None = None

    def _build_scanlines(self) -> pygame.Surface:
        surf = pygame.Surface((self._w, self._h), pygame.SRCALPHA)
        for y in range(0, self._h, 3):
            pygame.draw.line(surf, (0, 0, 0, 18), (0, y), (self._w - 1, y))
        return surf

    def _build_chevrons(self) -> list:
        size = 96
        result = []
        for corner in range(4):
            scratch = pygame.Surface((size, size), pygame.SRCALPHA)
            # Alternating diagonal stripes
            stripe_w = 12
            for i in range(-size, size * 2, stripe_w * 2):
                pts = [
                    (i, 0), (i + stripe_w, 0),
                    (i + stripe_w + size, size), (i + size, size),
                ]
                pygame.draw.polygon(scratch, (255, 176, 32), pts)
            # Dark fill chevrons on top
            for off in (0, 18):
                pts = [
                    (8,  size - 8 - off),
                    (size // 2, 12 + off),
                    (size - 8, size - 8 - off),
                    (size - 16, size - 8 - off),
                    (size // 2, 24 + off),
                    (16, size - 8 - off),
                ]
                pygame.draw.polygon(scratch, (28, 28, 28), pts)

            angle = [0, 90, 270, 180][corner]
            rotated = pygame.transform.rotate(scratch, angle)
            inset = 12
            positions = [
                (inset, inset),
                (self._w - inset - size, inset),
                (inset, self._h - inset - size),
                (self._w - inset - size, self._h - inset - size),
            ]
            result.append((rotated, positions[corner]))
        return result

    def update(self, radar_alive: bool, t_now: float) -> None:
        dt = t_now - self._last_t if self._last_t > 0 else 0.0
        self._last_t = t_now
        self._t_now  = t_now

        # Edge: healthy → failed
        if not radar_alive and self._was_alive:
            self._active    = True
            self._failure_t = t_now
            self._nominal_remaining = 0.0

        # Edge: failed → recovered
        if radar_alive and not self._was_alive:
            self._active    = False
            self._failure_t = None
            self._nominal_remaining = self.NOMINAL_PILL_S

        self._was_alive = radar_alive

        # Advance fade
        if self._active:
            self._fade_t = min(1.0, self._fade_t + dt / self.FADE_IN_S)
        else:
            self._fade_t = max(0.0, self._fade_t - dt / self.FADE_OUT_S)

        # Tick down nominal pill
        if self._nominal_remaining > 0:
            self._nominal_remaining = max(0.0, self._nominal_remaining - dt)

    def draw(self, screen: pygame.Surface) -> None:
        if self._fade_t == 0.0 and self._nominal_remaining == 0.0:
            return

        if self._fade_t > 0:
            self._draw_alert(screen)

        if self._nominal_remaining > 0 and self._fade_t == 0.0:
            self._draw_nominal_pill(screen)

    # ------------------------------------------------------------------ helpers

    def _pulse(self) -> float:
        """sin²-based pulse, always in [0.55, 1.0]."""
        s = math.sin(math.pi * self.BLINK_HZ * self._t_now)
        return 0.55 + 0.45 * s * s

    def _draw_alert(self, screen: pygame.Surface) -> None:
        fade = self._fade_t
        pulse = self._pulse()
        border_alpha = int(255 * fade * pulse)

        # 1. Double-rule animated red border
        outer_w, inner_w, gap = 12, 4, 2
        w, h = self._w, self._h
        for side in ("top", "bottom", "left", "right"):
            if side == "top":
                outer = pygame.Rect(0, 0, w, outer_w)
                inner = pygame.Rect(0, outer_w + gap, w, inner_w)
            elif side == "bottom":
                outer = pygame.Rect(0, h - outer_w, w, outer_w)
                inner = pygame.Rect(0, h - outer_w - gap - inner_w, w, inner_w)
            elif side == "left":
                outer = pygame.Rect(0, 0, outer_w, h)
                inner = pygame.Rect(outer_w + gap, 0, inner_w, h)
            else:
                outer = pygame.Rect(w - outer_w, 0, outer_w, h)
                inner = pygame.Rect(w - outer_w - gap - inner_w, 0, inner_w, h)

            outer_col = (*( 232, 58, 72), border_alpha)
            inner_col = (*(160, 22, 36), border_alpha)
            _blit_rect_alpha(screen, outer_col, outer)
            _blit_rect_alpha(screen, inner_col, inner)

        # 2. Corner chevrons — 180° out of phase with border
        chevron_alpha = int(255 * fade * (1.0 - pulse + 0.55))
        if self._chevrons is None:
            self._chevrons = self._build_chevrons()
        for surf, pos in self._chevrons:
            s = surf.copy()
            s.set_alpha(chevron_alpha)
            screen.blit(s, pos)

        # 3. Top center status banner
        self._draw_banner(screen, fade)

        # 4. Bottom-left mode pill
        self._draw_mode_pill(screen, fade)

        # 5. CRT scanlines
        if self._scanline_surf is None:
            self._scanline_surf = self._build_scanlines()
        sl = self._scanline_surf.copy()
        sl.set_alpha(int(28 * fade))
        screen.blit(sl, (0, 0))

    def _draw_banner(self, screen: pygame.Surface, fade: float) -> None:
        bw, bh = 480, 72
        bx = (self._w - bw) // 2
        by = 20
        banner = pygame.Surface((bw, bh), pygame.SRCALPHA)
        banner.fill((28, 8, 12, int(215 * fade)))
        pygame.draw.rect(banner, (232, 58, 72), (0, 0, bw, bh), 2)

        try:
            font_h1 = pygame.font.SysFont("consolas", 22, bold=True)
            font_h2 = pygame.font.SysFont("consolas", 16, bold=True)
            font_mono = pygame.font.SysFont("consolas", 13)
        except Exception:
            return

        # Line 1: ⚠ RADAR DOWN with blinking cursor
        cursor = " _" if int(self._t_now * 2) % 2 == 0 else "  "
        warn = "⚠"
        metrics = font_h1.metrics(warn)
        if not metrics or not metrics[0] or metrics[0][4] == 0:
            warn = "[!]"
        line1 = font_h1.render(f"{warn} RADAR DOWN{cursor}", True, (232, 58, 72))

        # Line 2
        line2 = font_h2.render("VISION-ONLY MODE", True, (255, 176, 32))

        # Line 3
        since = (self._t_now - self._failure_t) if self._failure_t else 0.0
        fail_at = self._failure_t if self._failure_t else 0.0
        line3 = font_mono.render(
            f"failover @ T+{fail_at:.1f}s   elapsed {since:.1f}s",
            True, (120, 134, 148),
        )

        banner.blit(line1, (bw // 2 - line1.get_width() // 2, 4))
        banner.blit(line2, (bw // 2 - line2.get_width() // 2, 28))
        banner.blit(line3, (bw // 2 - line3.get_width() // 2, 52))

        screen.blit(banner, (bx, by))

    def _draw_mode_pill(self, screen: pygame.Surface, fade: float) -> None:
        pw, ph = 260, 36
        px, py = 20, self._h - ph - 20
        pill = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pill.fill((*( 255, 176, 32), int(255 * fade)))
        try:
            font = pygame.font.SysFont("consolas", 13, bold=True)
            text = font.render("MODE: CV-LED  (FALLBACK)", True, (20, 16, 8))
            pill.blit(text, (pw // 2 - text.get_width() // 2, ph // 2 - text.get_height() // 2))
        except Exception:
            pass
        screen.blit(pill, (px, py))

    def _draw_nominal_pill(self, screen: pygame.Surface) -> None:
        pw, ph = 300, 36
        px, py = 20, self._h - ph - 20
        pill = pygame.Surface((pw, ph), pygame.SRCALPHA)
        # Fade out over last 0.5 s
        alpha = min(1.0, self._nominal_remaining / 0.5)
        pill.fill((*( 72, 210, 128), int(255 * alpha)))
        try:
            font = pygame.font.SysFont("consolas", 13, bold=True)
            text = font.render("RADAR NOMINAL  RESUMING RADAR-LED", True, (8, 28, 16))
            pill.blit(text, (pw // 2 - text.get_width() // 2, ph // 2 - text.get_height() // 2))
        except Exception:
            pass
        screen.blit(pill, (px, py))


def _blit_rect_alpha(
    screen: pygame.Surface,
    color_alpha: tuple,
    rect: pygame.Rect,
) -> None:
    r, g, b, a = color_alpha
    surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    surf.fill((r, g, b, a))
    screen.blit(surf, (rect.x, rect.y))
