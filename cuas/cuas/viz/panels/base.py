import pygame
from cuas.viz.theme import Theme


def draw_panel_chrome(
    surface: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    theme: Theme,
) -> None:
    pygame.draw.rect(surface, theme.panel_bg, rect)
    pygame.draw.rect(surface, theme.panel_border, rect, 1)
    label = theme.font_body.render(title, True, theme.text_dim)
    surface.blit(label, (rect.x + 6, rect.y + 4))
