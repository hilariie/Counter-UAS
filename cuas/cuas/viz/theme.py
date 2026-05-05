import pygame

# Palette (RGB)
BG           = (8,  10,  14)
PANEL_BG     = (14, 18,  24)
PANEL_BORDER = (40, 52,  68)
TEXT_PRIMARY = (220,228, 236)
TEXT_DIM     = (120,134, 148)
ACCENT_CYAN  = (64, 210, 232)
ACCENT_AMBER = (255,176,  32)
THREAT_RED   = (232, 58,  72)
CHOSEN_YELLOW= (244,212,  52)
EKF_GREEN    = (72, 210, 128)

# Layout rects (1280x720, 8px gutter) — populated after pygame.init()
WIDE        = None
NARROW      = None
THREAT_LIST = None
MINIMAP     = None
STATUS_STRIP= None


class Theme:
    def __init__(self):
        self.bg           = BG
        self.panel_bg     = PANEL_BG
        self.panel_border = PANEL_BORDER
        self.text_primary = TEXT_PRIMARY
        self.text_dim     = TEXT_DIM
        self.accent_cyan  = ACCENT_CYAN
        self.accent_amber = ACCENT_AMBER
        self.threat_red   = THREAT_RED
        self.chosen_yellow= CHOSEN_YELLOW
        self.ekf_green    = EKF_GREEN

        # fonts — populated by init()
        self.font_h1   = None  # size 22 bold
        self.font_h2   = None  # size 16 bold
        self.font_body = None  # size 14
        self.font_mono = None  # size 13

        # layout rects
        self.layout = {}

        # warning glyph
        self.warn_glyph = "⚠"

    def init(self):
        self.font_h1   = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_h2   = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_body = pygame.font.SysFont("consolas", 14)
        self.font_mono = pygame.font.SysFont("consolas", 13)

        # Detect ⚠ glyph support
        metrics = self.font_h1.metrics("⚠")
        if not metrics or metrics[0] is None or metrics[0][4] == 0:
            self.warn_glyph = "[!]"

        self.layout = {
            "WIDE":         pygame.Rect(  8,   8, 760, 432),
            "NARROW":       pygame.Rect(776,   8, 496, 280),
            "THREAT_LIST":  pygame.Rect(776, 296, 496, 240),
            "MINIMAP":      pygame.Rect(  8, 448, 760, 264),
            "STATUS_STRIP": pygame.Rect(776, 544, 496, 168),
        }


THEME_DEFAULT = Theme()
