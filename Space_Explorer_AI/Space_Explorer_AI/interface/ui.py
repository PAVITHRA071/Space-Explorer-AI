"""Small drawing helpers and widgets shared by all screens (Pygame only)."""

import math
import random

import pygame

import config

C = config.COLORS
_fonts = {}


def reset_fonts():
    """Forget cached fonts (they become invalid after ``pygame.quit()``)."""
    _fonts.clear()


def font(size, bold=False):
    """Pygame's built-in font: no font files are needed on any computer."""
    key = (size, bold)
    if key not in _fonts:
        created = pygame.font.Font(None, size)
        created.set_bold(bold)
        _fonts[key] = created
    return _fonts[key]


def draw_text(surface, text, size, color, pos, anchor="topleft", bold=False):
    image = font(size, bold).render(str(text), True, color)
    rect = image.get_rect(**{anchor: pos})
    surface.blit(image, rect)
    return rect


def wrap_text(text, size, max_width):
    """Split ``text`` into lines that fit into ``max_width`` pixels."""
    lines, current = [], ""
    for word in str(text).split():
        trial = word if not current else f"{current} {word}"
        if font(size).size(trial)[0] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_panel(surface, rect, fill=None, border=None, radius=12, alpha=None):
    rect = pygame.Rect(rect)
    fill = fill or C["panel"]
    if alpha is not None:
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(layer, (*fill, alpha), layer.get_rect(), border_radius=radius)
        surface.blit(layer, rect.topleft)
    else:
        pygame.draw.rect(surface, fill, rect, border_radius=radius)
    pygame.draw.rect(surface, border or C["panel_border"], rect, 2, border_radius=radius)


def draw_progress(surface, rect, fraction, color=None):
    rect = pygame.Rect(rect)
    fraction = max(0.0, min(1.0, fraction))
    pygame.draw.rect(surface, C["bg"], rect, border_radius=8)
    if fraction > 0:
        fill = pygame.Rect(rect.x, rect.y, max(8, int(rect.width * fraction)), rect.height)
        pygame.draw.rect(surface, color or C["cyan"], fill, border_radius=8)
    pygame.draw.rect(surface, C["panel_border"], rect, 2, border_radius=8)


class Starfield:
    """A twinkling star background."""

    def __init__(self, width, height, count=140, seed=7):
        rng = random.Random(seed)
        self.width, self.height = width, height
        self.stars = [(rng.randrange(width), rng.randrange(height),
                       rng.choice((1, 1, 1, 2)), rng.uniform(0, 6.28),
                       rng.uniform(0.6, 2.0)) for _ in range(count)]
        # A soft purple and blue glow so the background is not flat.
        self.base = pygame.Surface((width, height))
        self.base.fill(C["bg"])
        for center, radius, color in (((int(width * 0.85), int(height * 0.15)), 260, (40, 20, 90)),
                                      ((int(width * 0.1), int(height * 0.9)), 300, (10, 40, 90))):
            glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            for r in range(radius, 0, -12):
                alpha = int(34 * (1 - r / radius))
                pygame.draw.circle(glow, (*color, alpha), (radius, radius), r)
            self.base.blit(glow, (center[0] - radius, center[1] - radius))

    def draw(self, surface, t):
        surface.blit(self.base, (0, 0))
        for x, y, size, phase, speed in self.stars:
            level = 150 + int(90 * math.sin(phase + t * speed))
            level = max(60, min(255, level))
            pygame.draw.circle(surface, (level, level, min(255, level + 20)), (x, y), size)


class Button:
    def __init__(self, rect, label, callback, color=None, enabled=True):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.color = color or C["blue"]
        self.enabled = enabled

    def draw(self, surface, mouse_pos):
        hover = self.enabled and self.rect.collidepoint(mouse_pos)
        if not self.enabled:
            fill, border, text = (30, 36, 64), (50, 58, 96), (100, 110, 150)
        else:
            base = self.color
            fill = tuple(min(255, int(c * (0.55 if not hover else 0.8))) for c in base)
            border = base if not hover else C["white"]
            text = C["text"]
        pygame.draw.rect(surface, fill, self.rect, border_radius=10)
        pygame.draw.rect(surface, border, self.rect, 2, border_radius=10)
        draw_text(surface, self.label, 26, text, self.rect.center, "center", bold=True)

    def handle_event(self, event):
        if (event.type == pygame.MOUSEBUTTONUP and event.button == 1
                and self.enabled and self.rect.collidepoint(event.pos)):
            self.callback()
            return True
        return False


class TextInput:
    """A one-line text box for numbers."""

    ALLOWED = set("0123456789.-+eE")

    def __init__(self, rect, text=""):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.focused = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.focused = self.rect.collidepoint(event.pos)
            return self.focused
        if event.type == pygame.KEYDOWN and self.focused:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
                return True
            if event.unicode and event.unicode in self.ALLOWED and len(self.text) < 14:
                self.text += event.unicode
                return True
        return False

    def draw(self, surface, error=False):
        border = C["red"] if error else (C["cyan"] if self.focused else C["panel_border"])
        pygame.draw.rect(surface, C["bg"], self.rect, border_radius=6)
        pygame.draw.rect(surface, border, self.rect, 2, border_radius=6)
        draw_text(surface, self.text, 28, C["text"], (self.rect.x + 8, self.rect.centery), "midleft")
        if self.focused and (pygame.time.get_ticks() // 450) % 2 == 0:
            width = font(28).size(self.text)[0]
            x = self.rect.x + 10 + width
            pygame.draw.line(surface, C["cyan"], (x, self.rect.y + 6), (x, self.rect.bottom - 6), 2)


# --------------------------------------------------------------------------
# Game object drawings (simple shapes only; no image files are needed)
# --------------------------------------------------------------------------
def draw_ship(surface, center, size, direction, t):
    """A small spaceship pointing in ``direction`` = (dx, dy)."""
    dx, dy = direction
    angle = math.atan2(dx, -dy)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def place(x, y):
        return (center[0] + (x * cos_a - y * sin_a) * size,
                center[1] + (x * sin_a + y * cos_a) * size)

    flame = 0.16 + 0.07 * math.sin(t * 25)
    pygame.draw.polygon(surface, C["orange"],
                        [place(-0.12, 0.2), place(0.12, 0.2), place(0, 0.2 + flame + 0.14)])
    pygame.draw.polygon(surface, C["yellow"],
                        [place(-0.07, 0.2), place(0.07, 0.2), place(0, 0.2 + flame)])
    body = [place(0, -0.44), place(0.34, 0.34), place(0, 0.18), place(-0.34, 0.34)]
    pygame.draw.polygon(surface, C["cyan"], body)
    pygame.draw.polygon(surface, C["white"], body, 2)
    pygame.draw.circle(surface, C["bg"], place(0, -0.05), max(3, int(size * 0.1)))
    pygame.draw.circle(surface, C["blue"], place(0, -0.05), max(2, int(size * 0.06)))


def draw_asteroid(surface, center, size, seed):
    rng = random.Random(seed)
    points = []
    for i in range(10):
        angle = i / 10 * math.tau
        radius = size * 0.4 * rng.uniform(0.78, 1.12)
        points.append((center[0] + math.cos(angle) * radius,
                       center[1] + math.sin(angle) * radius))
    pygame.draw.polygon(surface, (120, 50, 36), points)
    pygame.draw.polygon(surface, C["orange"], points, 3)
    for _ in range(3):
        cx = center[0] + rng.uniform(-0.17, 0.17) * size
        cy = center[1] + rng.uniform(-0.17, 0.17) * size
        pygame.draw.circle(surface, (80, 30, 24), (int(cx), int(cy)), max(2, int(size * rng.uniform(0.05, 0.09))))


def draw_crystal(surface, center, size, t, phase=0.0):
    pulse = 0.5 + 0.5 * math.sin(t * 3 + phase)
    glow_radius = int(size * (0.36 + 0.08 * pulse))
    glow = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(glow, (255, 235, 90, 60 + int(50 * pulse)), (glow_radius, glow_radius), glow_radius)
    surface.blit(glow, (center[0] - glow_radius, center[1] - glow_radius))
    r = size * 0.27
    gem = [(center[0], center[1] - r * 1.25), (center[0] + r, center[1]),
           (center[0], center[1] + r * 1.25), (center[0] - r, center[1])]
    pygame.draw.polygon(surface, C["yellow"], gem)
    pygame.draw.polygon(surface, C["green"], [gem[0], gem[1], (center[0], center[1])])
    pygame.draw.polygon(surface, C["white"], gem, 2)


def draw_planet(surface, center, size, t):
    radius = int(size * 0.33)
    for i in range(radius, 0, -2):
        mix = i / radius
        color = (int(C["purple"][0] * mix + 40 * (1 - mix)),
                 int(C["purple"][1] * mix + 150 * (1 - mix)),
                 int(C["purple"][2] * mix + 255 * (1 - mix)))
        pygame.draw.circle(surface, color, (center[0] - int((1 - mix) * radius * 0.25),
                                             center[1] - int((1 - mix) * radius * 0.25)), i)
    ring = pygame.Rect(0, 0, int(size * 0.9), int(size * 0.26))
    ring.center = center
    pygame.draw.ellipse(surface, C["cyan"], ring, 3)
    pulse = int(3 + 2 * math.sin(t * 3))
    pygame.draw.circle(surface, C["green"], center, radius + pulse + 3, 2)
