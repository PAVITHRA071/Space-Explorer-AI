"""Comparison dashboard: Matplotlib graphs drawn into the Pygame window."""

import pygame

import config
from environment import SpaceEnvironment
from evaluation import plots
from evaluation.evaluate_agents import evaluate_q_table
from utils import storage

from . import ui

C = config.COLORS
GRAPH_RECT = pygame.Rect(12, 70, 1000, 500)
DPI = 100


class ComparisonScreen:
    def __init__(self, app):
        self.app = app
        self.data = {}
        self.problems = []
        self.message = None
        self.tab = "all"
        self.cache = {}
        self._load()
        self.buttons = [ui.Button((12, 590, 70, 40), "All", lambda: self.show("all"), C["purple"])]
        x = 92
        widths = {"reward": 170, "success": 150, "steps": 190, "summary": 120}
        for number, panel in enumerate(plots.PANELS, start=1):
            label = f"{number}. {plots.PANEL_LABELS[panel]}"
            width = widths[panel]
            self.buttons.append(ui.Button((x, 590, width, 40), label, lambda p=panel: self.show(p), C["blue"]))
            x += width + 10
        # Last buttons end at x = 1002, inside the 1024 px wide window.
        self.buttons.append(ui.Button((x, 590, 120, 40), "Save PNG", self.save_png, C["green"]))
        self.buttons.append(ui.Button((x + 130, 590, 110, 40), "Menu", app.go_menu, (120, 60, 90)))

    def _load(self):
        env = SpaceEnvironment()
        for algorithm in config.ALGORITHM_KEYS:
            name = config.ALGORITHM_NAMES[algorithm]
            try:
                model = storage.load_model(algorithm, env.signature(), env.n_states, env.n_actions)
            except storage.ModelError as error:
                self.problems.append(str(error))
                continue
            # Fresh evaluation with the fixed evaluation policy (not training behaviour).
            evaluation = evaluate_q_table(model.q_table, env)
            self.data[algorithm] = {
                "history": model.history,
                "evaluation": evaluation,
                "episodes": int(model.meta.get("episodes", len(model.history["reward"]))),
            }
            try:
                storage.update_evaluation_summary(algorithm, evaluation, self.data[algorithm]["episodes"],
                                                  model.meta.get("seed"))
            except OSError as error:
                self.problems.append(f"Could not update the results CSV for {name}: {error}")

    def show(self, tab):
        self.tab = tab

    def _surface(self):
        if self.tab not in self.cache:
            fig = plots.build_figure(self.data, self.tab,
                                     figsize=(GRAPH_RECT.width / DPI, GRAPH_RECT.height / DPI), dpi=DPI)
            raw, size = plots.figure_to_rgba(fig)
            self.cache[self.tab] = pygame.image.frombuffer(raw, size, "RGBA").convert()
        return self.cache[self.tab]

    def save_png(self):
        if not self.data:
            self.message = ("Nothing to save yet: train an algorithm first.", C["orange"])
            return
        path = config.RESULTS_DIR / "comparison.png"
        try:
            config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            plots.save_comparison_png(self.data, path)
            self.message = ("Saved results/comparison.png (CSV files are in results/ too).", C["green"])
        except OSError as error:
            self.message = (f"Could not save the picture: {error}", C["red"])

    def handle_event(self, event):
        for button in self.buttons:
            if button.handle_event(event):
                return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.go_menu()
            elif event.key == pygame.K_0 or event.key == pygame.K_a:
                self.show("all")
            elif pygame.K_1 <= event.key <= pygame.K_4:
                self.show(plots.PANELS[event.key - pygame.K_1])

    def update(self, dt):
        pass

    def draw(self, surface):
        ui.draw_text(surface, "Compare Algorithms", 44, C["cyan"], (16, 16), bold=True)
        if not self.data:
            ui.draw_panel(surface, (60, 120, 900, 300))
            ui.draw_text(surface, "No trained models to compare yet.", 40, C["yellow"], (90, 150), bold=True)
            y = 215
            for line in self.problems:
                for part in ui.wrap_text(line, 26, 840):
                    ui.draw_text(surface, part, 26, C["muted"], (90, y))
                    y += 28
            ui.draw_text(surface, "Choose 'Train Both Algorithms' in the main menu, then come back.",
                         26, C["text"], (90, y + 20))
        else:
            surface.blit(self._surface(), GRAPH_RECT.topleft)
            if self.problems:
                ui.draw_text(surface, self.problems[0], 22, C["orange"], (420, 30))
        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            button.draw(surface, mouse)
        if self.message:
            text, color = self.message
            ui.draw_text(surface, text, 24, color, (16, 640))
