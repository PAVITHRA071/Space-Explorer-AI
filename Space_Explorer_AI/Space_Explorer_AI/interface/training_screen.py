"""Training screen: trains in small time slices so the window never freezes."""

import random

import pygame

import config
from environment import SpaceEnvironment
from training.trainer import Trainer, finalize_training
from evaluation.evaluate_agents import format_steps

from . import ui

C = config.COLORS
TIME_SLICE = 0.012  # seconds of training per frame (keeps ~60 fps)


class TrainingScreen:
    def __init__(self, app, algorithms):
        self.app = app
        self.algorithms = list(algorithms)
        self.index = 0
        self.state = "training"      # training | done | cancelled | error
        self.error = None
        self.summaries = []          # (algorithm, evaluation, episodes)
        seed = app.settings.seed
        self.seed = random.randrange(2**31) if seed == -1 else seed
        self.chart = []
        self.chart_step = 1
        self.trainer = None
        self._start_trainer()
        self.buttons = []
        self._make_buttons()

    def _start_trainer(self):
        algorithm = self.algorithms[self.index]
        self.env = SpaceEnvironment()
        # The same seed for every algorithm gives comparable starting conditions.
        self.trainer = Trainer(algorithm, self.app.settings, seed=self.seed, env=self.env)
        self.chart = []
        self.chart_step = max(1, self.trainer.total_episodes // 300)

    def _make_buttons(self):
        app = self.app
        self.buttons = []
        if self.state == "training":
            self.buttons.append(ui.Button((412, 575, 200, 46), "Stop (Esc)", self.stop, (170, 70, 70)))
            return
        x = 60
        if self.state == "done":
            self.buttons.append(ui.Button((x, 575, 230, 46), "Compare Algorithms", app.start_compare, C["yellow"]))
            x += 245
            for algorithm in self.algorithms:
                name = config.ALGORITHM_NAMES[algorithm]
                self.buttons.append(ui.Button((x, 575, 230, 46), f"Watch {name}",
                                              lambda a=algorithm: app.start_watch(a), C["cyan"]))
                x += 245
        self.buttons.append(ui.Button((800, 575, 170, 46), "Menu (Esc)", app.go_menu, (120, 60, 90)))

    def stop(self):
        if self.state == "training":
            self.trainer.stop()
            self.state = "cancelled"
            self._make_buttons()

    def handle_event(self, event):
        for button in self.buttons:
            if button.handle_event(event):
                return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.state == "training":
                self.stop()
            else:
                self.app.go_menu()

    def update(self, dt):
        if self.state != "training":
            return
        before = self.trainer.episode
        self.trainer.run_episodes(time_budget=TIME_SLICE)
        if self.trainer.episode // self.chart_step > before // self.chart_step:
            self.chart.append(self.trainer.snapshot()["avg_reward"])
        if self.trainer.finished:
            self._finish_current()

    def _finish_current(self):
        result = self.trainer.result()
        try:
            evaluation = finalize_training(result, self.env)
        except OSError as error:
            self.state, self.error = "error", f"Could not save the results: {error}"
            self._make_buttons()
            return
        self.summaries.append((result.algorithm, evaluation, result.episodes_completed))
        self.index += 1
        if self.index < len(self.algorithms):
            self._start_trainer()
        else:
            self.state = "done"
            self._make_buttons()

    # ------------------------------------------------------------------
    def draw(self, surface):
        title = "Training" if self.state == "training" else {
            "done": "Training complete", "cancelled": "Training stopped", "error": "Error"}[self.state]
        ui.draw_text(surface, title, 56, C["cyan"], (60, 24), bold=True)
        if self.state == "training" or self.state == "cancelled":
            self._draw_progress(surface)
        if self.summaries:
            self._draw_summaries(surface)
        if self.state == "cancelled":
            ui.draw_text(surface, "Training was stopped. Nothing was saved for the unfinished run; "
                         "your previously saved model is unchanged.", 24, C["orange"], (60, 545))
        if self.state == "error":
            ui.draw_text(surface, self.error, 26, C["red"], (60, 528))
        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            button.draw(surface, mouse)

    def _draw_progress(self, surface):
        snap = self.trainer.snapshot()
        name = config.ALGORITHM_NAMES[snap["algorithm"]]
        count = len(self.algorithms)
        label = f"{name}" + (f"   (algorithm {min(self.index + 1, count)} of {count})" if count > 1 else "")
        ui.draw_text(surface, label, 36, C["yellow"], (60, 90), bold=True)
        panel = pygame.Rect(60, 135, 420, 250)
        ui.draw_panel(surface, panel)
        rows = [
            ("Episode", f"{snap['episode']} / {snap['total']}"),
            ("Exploration rate (epsilon)", f"{snap['epsilon']:.3f}"),
            ("Last episode reward", f"{snap['last_reward']:.0f}"),
            (f"Average reward (last {config.RECENT_WINDOW})", f"{snap['avg_reward']:.1f}"),
            (f"Success rate (last {config.RECENT_WINDOW})", f"{snap['success_rate']:.0f}%"),
            ("Progress", f"{snap['progress'] * 100:.0f}%"),
        ]
        y = panel.y + 16
        for label_text, value in rows:
            ui.draw_text(surface, label_text, 24, C["muted"], (panel.x + 16, y))
            ui.draw_text(surface, value, 28, C["text"], (panel.right - 16, y - 2), "topright", bold=True)
            y += 38
        ui.draw_progress(surface, (60, 400, 420, 26), snap["progress"])
        if count > 1:
            overall = (self.index + snap["progress"]) / count
            ui.draw_text(surface, "Overall", 22, C["muted"], (60, 436))
            ui.draw_progress(surface, (140, 436, 340, 20), overall, C["purple"])
        ui.draw_text(surface, f"Random seed: {self.seed}", 22, C["muted"], (60, 470))

        # Live chart of the rolling average reward
        chart = pygame.Rect(520, 135, 450, 250)
        ui.draw_panel(surface, chart)
        ui.draw_text(surface, "Average reward while learning", 24, C["text"], (chart.x + 14, chart.y + 10))
        inner = pygame.Rect(chart.x + 16, chart.y + 46, chart.width - 32, chart.height - 64)
        pygame.draw.rect(surface, C["bg"], inner)
        if len(self.chart) >= 2:
            low, high = min(self.chart), max(self.chart)
            span = max(1e-9, high - low)
            points = [(inner.x + i / (len(self.chart) - 1) * inner.width,
                       inner.bottom - (v - low) / span * (inner.height - 6) - 3)
                      for i, v in enumerate(self.chart)]
            pygame.draw.lines(surface, config_color(snap["algorithm"]), False, points, 2)
            ui.draw_text(surface, f"{high:.0f}", 18, C["muted"], (inner.x + 4, inner.y + 2))
            ui.draw_text(surface, f"{low:.0f}", 18, C["muted"], (inner.x + 4, inner.bottom - 16))
        else:
            ui.draw_text(surface, "collecting data...", 22, C["muted"], inner.center, "center")

    def _draw_summaries(self, surface):
        top = 100 if self.state in ("done", "error") else 502
        ui.draw_text(surface, "Evaluation of finished models (greedy policy):", 24, C["text"], (60, top))
        y = top + 28
        for algorithm, ev, episodes in self.summaries:
            name = config.ALGORITHM_NAMES[algorithm]
            text = (f"{name}: {episodes} episodes | mean reward {ev['mean_reward']:.1f} | "
                    f"success {ev['success_rate']:.0f}% | steps/success {format_steps(ev['mean_steps_success'])} | "
                    f"collisions {ev['collision_rate']:.0f}% | crystals {ev['mean_crystals']:.2f}")
            ui.draw_text(surface, text, 22, C["muted"], (60, y))
            y += 26
        if self.state == "done":
            ui.draw_text(surface, "Models and CSV files were saved in the models/ and results/ folders.",
                         24, C["green"], (60, y + 12))


def config_color(algorithm):
    """Pygame colour for an algorithm (the config stores hex strings for Matplotlib)."""
    return pygame.Color(config.ALGORITHM_COLORS[algorithm])
