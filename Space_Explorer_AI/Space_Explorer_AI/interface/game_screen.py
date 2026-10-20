"""The game screen: human play and trained-agent playback.

Both modes use exactly the same ``SpaceEnvironment`` and reward rules.
"""

import numpy as np
import pygame

import config
from environment import (ACTION_NAMES, DOWN, LEFT, RIGHT, UP, STATUS_CRASHED,
                         STATUS_RUNNING, STATUS_SUCCESS, STATUS_TIMEOUT, SpaceEnvironment)
from environment.space_environment import ACTION_DELTAS

from . import ui

C = config.COLORS
STATUS_TEXT = {
    STATUS_RUNNING: ("Mission in progress", C["cyan"]),
    STATUS_SUCCESS: ("MISSION SUCCESS", C["green"]),
    STATUS_CRASHED: ("CRASHED INTO ASTEROID", C["red"]),
    STATUS_TIMEOUT: ("OUT OF FUEL", C["orange"]),
}
HUMAN_KEYS = {
    pygame.K_UP: UP, pygame.K_w: UP,
    pygame.K_DOWN: DOWN, pygame.K_s: DOWN,
    pygame.K_LEFT: LEFT, pygame.K_a: LEFT,
    pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
}


def cell_center(env_pos, origin=config.BOARD_ORIGIN, cell=config.CELL_SIZE):
    return (origin[0] + env_pos[0] * cell + cell // 2,
            origin[1] + env_pos[1] * cell + cell // 2)


def draw_board(surface, env, t, ship_direction, origin=config.BOARD_ORIGIN, cell=config.CELL_SIZE):
    """Draw the grid and every object of ``env``."""
    width, height = env.cols * cell, env.rows * cell
    ui.draw_panel(surface, (origin[0] - 6, origin[1] - 6, width + 12, height + 12),
                  fill=C["bg_light"], radius=10)
    for y in range(env.rows):
        for x in range(env.cols):
            rect = pygame.Rect(origin[0] + x * cell, origin[1] + y * cell, cell, cell)
            pygame.draw.rect(surface, C["cell"] if (x + y) % 2 == 0 else C["cell_alt"], rect)
            pygame.draw.rect(surface, C["grid_line"], rect, 1)

    # Start marker
    start_center = cell_center(env.start, origin, cell)
    pygame.draw.circle(surface, C["blue"], start_center, cell // 2 - 8, 2)
    ui.draw_text(surface, "START", 16, C["muted"], (start_center[0], start_center[1] + cell // 2 - 12), "center")

    for index, pos in enumerate(sorted(env.asteroids)):
        ui.draw_asteroid(surface, cell_center(pos, origin, cell), cell, seed=pos[0] * 31 + pos[1])
    for index, pos in enumerate(env.remaining_crystals()):
        ui.draw_crystal(surface, cell_center(pos, origin, cell), cell, t, phase=index)
    ui.draw_planet(surface, cell_center(env.goal, origin, cell), cell, t)
    ui.draw_ship(surface, cell_center(env.position, origin, cell), cell * 0.9, ship_direction, t)


class GameScreen:
    """``mode`` is "human" or "ai"; for "ai" a Q-table and algorithm are given."""

    def __init__(self, app, mode, q_table=None, algorithm=None):
        self.app = app
        self.mode = mode
        self.q_table = q_table
        self.algorithm = algorithm
        self.env = SpaceEnvironment()
        self.speed = app.settings.playback_speed
        self.paused = False
        self.timer = 0.0
        self.t = 0.0
        self.ship_direction = ACTION_DELTAS[RIGHT]
        self.last_action = None
        self.popups = []   # (text, color, position, age)
        self.title = ("Human Play" if mode == "human"
                      else f"{config.ALGORITHM_NAMES[algorithm]} Agent (greedy policy)")

        px, py, pw, _ = config.PANEL_RECT
        self.buttons = [
            ui.Button((px + 16, py + 340, 135, 38), "Pause (P)", self.toggle_pause, C["blue"]),
            ui.Button((px + 161, py + 340, 135, 38), "Restart (R)", self.restart, C["purple"]),
            ui.Button((px + 16, py + 458, pw - 32, 38), "Menu (Esc)", app.go_menu, (120, 60, 90)),
        ]
        self.pause_button = self.buttons[0]
        if mode == "ai":
            self.buttons.append(ui.Button((px + 16, py + 388, 135, 38), "Slower (-)", self.slower, C["blue"]))
            self.buttons.append(ui.Button((px + 161, py + 388, 135, 38), "Faster (+)", self.faster, C["blue"]))

    # ------------------------------------------------------------------
    def toggle_pause(self):
        self.paused = not self.paused

    def restart(self):
        self.env.reset()
        self.paused = False
        self.timer = 0.0
        self.last_action = None
        self.popups.clear()

    def slower(self):
        self.speed = max(1, self.speed - 1)

    def faster(self):
        self.speed = min(30, self.speed + 1)

    def _move(self, action):
        if self.env.done:
            return
        _, reward, _, info = self.env.step(action)
        self.last_action = action
        self.ship_direction = ACTION_DELTAS[action]
        pos = cell_center(self.env.position)
        if info["crystal_collected"]:
            self.popups.append((f"+{self.env.rewards['crystal']}", C["yellow"], pos, 0.0))
        elif info["terminated"]:
            text = f"+{reward}" if reward > 0 else str(reward)
            color = C["green"] if reward > 0 else C["red"]
            self.popups.append((text, color, pos, 0.0))

    # ------------------------------------------------------------------
    def handle_event(self, event):
        for button in self.buttons:
            if button.handle_event(event):
                return
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE or event.key == pygame.K_m:
            self.app.go_menu()
        elif event.key == pygame.K_p or (self.mode == "ai" and event.key == pygame.K_SPACE):
            self.toggle_pause()
        elif event.key == pygame.K_r or (self.env.done and event.key == pygame.K_RETURN):
            self.restart()
        elif self.mode == "human":
            if event.key in HUMAN_KEYS and not self.paused:
                self._move(HUMAN_KEYS[event.key])
        else:
            if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.slower()
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.faster()

    def update(self, dt):
        self.t += dt
        self.popups = [(txt, col, pos, age + dt) for txt, col, pos, age in self.popups if age < 1.0]
        self.pause_button.label = "Resume (P)" if self.paused else "Pause (P)"
        if self.mode == "ai" and not self.paused and not self.env.done:
            self.timer += dt
            interval = 1.0 / self.speed
            while self.timer >= interval and not self.env.done:
                self.timer -= interval
                state = self.env.state_to_index()
                # Greedy policy: highest Q-value; argmax breaks ties the same way every time.
                self._move(int(np.argmax(self.q_table[state])))

    # ------------------------------------------------------------------
    def draw(self, surface):
        ui.draw_text(surface, "SPACE EXPLORER AI", 52, C["cyan"], (24, 22), bold=True)
        ui.draw_text(surface, self.title, 30, C["muted"], (26, 68))
        draw_board(surface, self.env, self.t, self.ship_direction)
        for text, color, pos, age in self.popups:
            ui.draw_text(surface, text, 34, color, (pos[0], pos[1] - 20 - age * 40), "center", bold=True)

        px, py, pw, ph = config.PANEL_RECT
        ui.draw_panel(surface, config.PANEL_RECT)
        env = self.env
        rows = [
            ("Mode", "Human" if self.mode == "human" else config.ALGORITHM_NAMES[self.algorithm]),
            ("Score", f"{env.total_reward}"),
            ("Position", f"({env.position[0]}, {env.position[1]})"),
            ("Moves", f"{env.steps} / {env.max_steps}"),
            ("Crystals", f"{env.crystals_collected} / {env.n_crystals}"),
            ("Last move", ACTION_NAMES[self.last_action] if self.last_action is not None else "-"),
        ]
        if self.mode == "ai":
            rows.append(("Speed", f"{self.speed} steps/s"))
        y = py + 14
        for label, value in rows:
            ui.draw_text(surface, label, 26, C["muted"], (px + 18, y))
            ui.draw_text(surface, value, 28, C["text"], (px + pw - 18, y), "topright", bold=True)
            y += 32
        text, color = STATUS_TEXT[env.status]
        if self.paused and env.status == STATUS_RUNNING:
            text, color = "PAUSED", C["yellow"]
        ui.draw_text(surface, text, 30, color, (px + pw // 2, py + 296), "center", bold=True)
        for button in self.buttons:
            button.draw(surface, pygame.mouse.get_pos())
        hint = ("Arrow keys / WASD to fly" if self.mode == "human"
                else "Space: play/pause   -/+: speed")
        ui.draw_text(surface, hint, 22, C["muted"], (px + pw // 2, py + 436 if self.mode == "human" else py + 440), "center")

        if env.done:
            self._draw_result_overlay(surface)

    def _draw_result_overlay(self, surface):
        origin = config.BOARD_ORIGIN
        width, height = self.env.cols * config.CELL_SIZE, self.env.rows * config.CELL_SIZE
        rect = pygame.Rect(0, 0, 420, 150)
        rect.center = (origin[0] + width // 2, origin[1] + height // 2)
        ui.draw_panel(surface, rect, fill=C["bg"], alpha=235)
        text, color = STATUS_TEXT[self.env.status]
        ui.draw_text(surface, text, 44, color, (rect.centerx, rect.y + 42), "center", bold=True)
        summary = (f"Score {self.env.total_reward}   Moves {self.env.steps}   "
                   f"Crystals {self.env.crystals_collected}/{self.env.n_crystals}")
        ui.draw_text(surface, summary, 26, C["text"], (rect.centerx, rect.y + 86), "center")
        ui.draw_text(surface, "Press R or Enter to play again", 24, C["muted"],
                     (rect.centerx, rect.y + 120), "center")
