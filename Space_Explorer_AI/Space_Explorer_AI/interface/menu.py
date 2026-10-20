"""Main menu and settings screens."""

import pygame

import config
from utils import storage

from . import ui

C = config.COLORS


class MenuScreen:
    def __init__(self, app, message=None):
        self.app = app
        self.t = 0.0
        self.message = message  # (text, color) or None
        self.models = {key: storage.model_info(key) for key in config.ALGORITHM_KEYS}
        items = [
            ("Start Human Play", app.start_human, C["green"]),
            ("Train Q-Learning", lambda: app.start_training([config.Q_LEARNING]), C["blue"]),
            ("Train SARSA", lambda: app.start_training([config.SARSA]), C["blue"]),
            ("Train Both Algorithms", lambda: app.start_training(list(config.ALGORITHM_KEYS)), C["purple"]),
            ("Watch Q-Learning Agent", lambda: app.start_watch(config.Q_LEARNING), C["cyan"]),
            ("Watch SARSA Agent", lambda: app.start_watch(config.SARSA), C["cyan"]),
            ("Compare Algorithms", app.start_compare, C["yellow"]),
            ("Settings", app.start_settings, (110, 120, 170)),
            ("Exit", app.quit, (150, 60, 70)),
        ]
        self.buttons = []
        self.hotkeys = {}
        for index, (label, action, color) in enumerate(items):
            rect = (80, 150 + index * 50, 400, 42)
            self.buttons.append(ui.Button(rect, f"{index + 1}  {label}", action, color))
            self.hotkeys[pygame.K_1 + index] = action

    def handle_event(self, event):
        for button in self.buttons:
            if button.handle_event(event):
                return
        if event.type == pygame.KEYDOWN:
            if event.key in self.hotkeys:
                self.hotkeys[event.key]()
            elif event.key == pygame.K_ESCAPE:
                self.app.quit()

    def update(self, dt):
        self.t += dt

    def draw(self, surface):
        ui.draw_text(surface, "SPACE EXPLORER AI", 76, C["cyan"], (80, 34), bold=True)
        ui.draw_text(surface, "Q-Learning vs SARSA: learning to navigate space", 28, C["muted"], (84, 100))
        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            button.draw(surface, mouse)

        panel = pygame.Rect(540, 150, 440, 230)
        ui.draw_panel(surface, panel)
        ui.draw_text(surface, "Saved models", 30, C["yellow"], (panel.x + 18, panel.y + 14), bold=True)
        y = panel.y + 56
        for key in config.ALGORITHM_KEYS:
            name = config.ALGORITHM_NAMES[key]
            info = self.models[key]
            if info is None:
                status, color = "not trained yet", C["muted"]
            elif "error" in info:
                status, color = "unreadable - train again", C["red"]
            else:
                status = f"{info.get('episodes', '?')} episodes ({info.get('saved_at', '')[:10]})"
                color = C["green"]
            ui.draw_text(surface, name, 28, C["text"], (panel.x + 18, y), bold=True)
            ui.draw_text(surface, status, 24, color, (panel.x + 18, y + 28))
            y += 70
        ui.draw_text(surface, f"Default training: {self.app.settings.episodes} episodes",
                     22, C["muted"], (panel.x + 18, panel.bottom - 30))

        # Decoration: a ship flying towards the planet.
        ui.draw_planet(surface, (880, 500), 150, self.t)
        ui.draw_ship(surface, (640, 470 + int(8 * pygame.math.Vector2(0, 1).y * 0) ), 80, (1, -0.3), self.t)
        ui.draw_crystal(surface, (740, 440), 60, self.t)

        if self.message:
            text, color = self.message
            for i, line in enumerate(ui.wrap_text(text, 26, 940)):
                ui.draw_text(surface, line, 26, color, (80, 612 + i * 26))
        else:
            ui.draw_text(surface, "Press 1-9 or click a button.", 24, C["muted"], (80, 618))


class SettingsScreen:
    ORDER = ("episodes", "alpha", "gamma", "epsilon_start",
             "epsilon_decay", "epsilon_min", "playback_speed", "seed")

    def __init__(self, app):
        self.app = app
        self.message = None
        self.errors = set()
        self.inputs = {}
        settings = app.settings
        for index, name in enumerate(self.ORDER):
            column, row = divmod(index, 4)
            x = 60 + column * 490
            y = 150 + row * 92
            rect = (x + 260, y, 160, 36)
            self.inputs[name] = (ui.TextInput(rect, format(getattr(settings, name), "g")), (x, y))
        self.buttons = [
            ui.Button((60, 560, 190, 44), "Save (Enter)", self.save, C["green"]),
            ui.Button((270, 560, 230, 44), "Reset defaults", self.reset, C["purple"]),
            ui.Button((780, 560, 190, 44), "Back (Esc)", app.go_menu, (120, 60, 90)),
        ]

    def reset(self):
        defaults = config.Settings()
        for name, (box, _) in self.inputs.items():
            box.text = format(getattr(defaults, name), "g")
        self.errors.clear()
        self.message = ("Defaults restored (press Save to keep them).", C["yellow"])

    def save(self):
        values, errors, problems = {}, set(), []
        for name, (box, _) in self.inputs.items():
            try:
                values[name] = config.parse_field(name, box.text)
            except ValueError as error:
                errors.add(name)
                problems.append(str(error))
        self.errors = errors
        if not problems:
            candidate = config.Settings(**values)
            problems = candidate.problems()
            if problems:
                self.errors = {"epsilon_start", "epsilon_min"}
        if problems:
            self.message = (problems[0], C["red"])
            return
        self.app.settings = candidate
        try:
            storage.save_settings(candidate)
            self.message = ("Settings saved.", C["green"])
        except OSError as error:
            self.message = (f"Settings applied but could not be saved to disk: {error}", C["orange"])

    def handle_event(self, event):
        for button in self.buttons:
            if button.handle_event(event):
                return
        for box, _ in self.inputs.values():
            box.handle_event(event)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.go_menu()
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.save()
            elif event.key == pygame.K_TAB:
                boxes = [box for box, _ in self.inputs.values()]
                focused = next((i for i, box in enumerate(boxes) if box.focused), -1)
                for box in boxes:
                    box.focused = False
                boxes[(focused + 1) % len(boxes)].focused = True

    def update(self, dt):
        pass

    def draw(self, surface):
        ui.draw_text(surface, "Settings", 60, C["cyan"], (60, 30), bold=True)
        ui.draw_text(surface, "Click a box and type. Tab moves to the next box.", 26, C["muted"], (62, 86))
        for name, (box, (x, y)) in self.inputs.items():
            label, kind, low, high = config.FIELD_SPECS[name]
            ui.draw_text(surface, label, 26, C["text"], (x, y + 2))
            allowed = f"{low} to {high}" if kind is int else f"{low:g} to {high:g}"
            ui.draw_text(surface, f"allowed: {allowed}", 20, C["muted"], (x, y + 30))
            box.draw(surface, error=name in self.errors)
        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            button.draw(surface, mouse)
        if self.message:
            text, color = self.message
            for i, line in enumerate(ui.wrap_text(text, 26, 900)):
                ui.draw_text(surface, line, 26, color, (60, 516 + i * 26))
