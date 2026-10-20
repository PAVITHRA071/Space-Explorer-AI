"""Starts the real Pygame application without a visible window (dummy video driver)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None

import config  # noqa: E402


@unittest.skipIf(pygame is None, "pygame is not installed")
class InterfaceSmokeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        # Keep test files away from the real models/ and results/ folders.
        self._patches = [
            mock.patch.object(config, "MODELS_DIR", root / "models"),
            mock.patch.object(config, "RESULTS_DIR", root / "results"),
            mock.patch.object(config, "SETTINGS_PATH", root / "settings.json"),
        ]
        for patch in self._patches:
            patch.start()
        import main
        self.app = main.App()
        self.app.settings.episodes = 40

    def tearDown(self):
        pygame.quit()
        for patch in self._patches:
            patch.stop()
        self._tmp.cleanup()

    def frames(self, count=3, dt=0.05):
        for _ in range(count):
            self.app.process_frame(dt)

    def key(self, key, unicode=""):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=unicode, mod=0))

    def test_menu_human_play_and_back(self):
        self.frames()
        self.key(pygame.K_1, "1")
        self.frames()
        self.assertEqual(type(self.app.screen).__name__, "GameScreen")
        for key in (pygame.K_UP, pygame.K_RIGHT, pygame.K_d, pygame.K_w):
            self.key(key)
        self.frames()
        self.assertEqual(self.app.screen.env.steps, 4)
        self.key(pygame.K_r)
        self.frames()
        self.assertEqual(self.app.screen.env.steps, 0)
        self.key(pygame.K_ESCAPE)
        self.frames()
        self.assertEqual(type(self.app.screen).__name__, "MenuScreen")

    def test_watch_without_model_shows_message(self):
        self.app.start_watch(config.Q_LEARNING)
        self.frames()
        self.assertEqual(type(self.app.screen).__name__, "MenuScreen")
        self.assertIn("Train", self.app.screen.message[0])

    def test_train_both_then_watch_and_compare(self):
        self.app.start_training(list(config.ALGORITHM_KEYS))
        for _ in range(400):
            self.app.process_frame(0.02)
            if self.app.screen.state != "training":
                break
        self.assertEqual(self.app.screen.state, "done")
        self.assertEqual(len(self.app.screen.summaries), 2)
        self.frames()
        self.app.start_watch(config.SARSA)
        for _ in range(60):
            self.app.process_frame(1.0)  # ~4 moves per frame; an episode has at most 100
        self.assertTrue(self.app.screen.env.done)
        self.app.start_compare()
        self.assertEqual(len(self.app.screen.data), 2)
        for tab in ("all", "reward", "success", "steps", "summary"):
            self.app.screen.show(tab)
            self.frames(1)
        self.app.screen.save_png()
        self.assertTrue((config.RESULTS_DIR / "comparison.png").exists())

    def test_stop_training_keeps_old_model(self):
        self.app.start_training([config.Q_LEARNING])
        self.app.settings.episodes = 40
        self.app.screen.stop()
        self.frames()
        self.assertEqual(self.app.screen.state, "cancelled")
        self.assertFalse((config.MODELS_DIR / "q_learning_model.npz").exists())

    def test_settings_validation(self):
        self.app.start_settings()
        screen = self.app.screen
        screen.inputs["episodes"][0].text = "abc"
        screen.save()
        self.assertIn("episodes", screen.errors)
        screen.inputs["episodes"][0].text = "120"
        screen.save()
        self.assertEqual(self.app.settings.episodes, 120)
        self.assertTrue(config.SETTINGS_PATH.exists())
        self.frames()

    def test_compare_without_models_explains(self):
        self.app.start_compare()
        self.frames()
        self.assertEqual(self.app.screen.data, {})


if __name__ == "__main__":
    unittest.main()
