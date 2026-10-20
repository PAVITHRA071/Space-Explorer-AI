"""Training, evaluation, storage and graph generation (small episode counts)."""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from environment import SpaceEnvironment  # noqa: E402
from evaluation import plots  # noqa: E402
from evaluation.evaluate_agents import evaluate_q_table, format_steps  # noqa: E402
from training.trainer import (Trainer, finalize_training, run_multi_seed_experiment,  # noqa: E402
                              train, train_and_save)
from utils import storage  # noqa: E402


def small_settings(episodes=60):
    settings = config.Settings()
    settings.episodes = episodes
    return settings


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.models = Path(self._tmp.name) / "models"
        self.results = Path(self._tmp.name) / "results"

    def tearDown(self):
        self._tmp.cleanup()

    def test_training_runs_for_both_algorithms(self):
        for algorithm in config.ALGORITHM_KEYS:
            result = train(algorithm, small_settings(), seed=1)
            self.assertEqual(result.episodes_completed, 60)
            self.assertFalse(result.cancelled)
            self.assertEqual(len(result.history["reward"]), 60)
            self.assertTrue(np.any(result.agent.q_table != 0))

    def test_training_is_reproducible_with_seed(self):
        a = train("sarsa", small_settings(), seed=9)
        b = train("sarsa", small_settings(), seed=9)
        np.testing.assert_array_equal(a.agent.q_table, b.agent.q_table)

    def test_stop_cancels_safely(self):
        trainer = Trainer("q_learning", small_settings(1000), seed=1)
        trainer.run_episodes(count=10)
        trainer.stop()
        self.assertTrue(trainer.finished)
        self.assertEqual(trainer.run_episodes(count=10), 0)
        result = trainer.result()
        self.assertTrue(result.cancelled)
        with self.assertRaises(ValueError):
            finalize_training(result, trainer.env, self.models, self.results)

    def test_save_and_load_model(self):
        result, evaluation = train_and_save("q_learning", small_settings(), seed=3,
                                            models_dir=self.models, results_dir=self.results)
        env = SpaceEnvironment()
        loaded = storage.load_model("q_learning", env.signature(), env.n_states,
                                    env.n_actions, self.models)
        np.testing.assert_array_equal(loaded.q_table, result.agent.q_table)
        self.assertEqual(loaded.meta["episodes"], 60)
        self.assertEqual(loaded.meta["seed"], 3)
        self.assertEqual(len(loaded.history["reward"]), 60)
        self.assertTrue((self.results / "training_history_q_learning.csv").exists())

    def test_missing_and_incompatible_models_give_clear_errors(self):
        env = SpaceEnvironment()
        with self.assertRaises(storage.ModelNotFoundError):
            storage.load_model("sarsa", env.signature(), env.n_states, env.n_actions, self.models)
        train_and_save("sarsa", small_settings(), seed=1,
                       models_dir=self.models, results_dir=self.results)
        with self.assertRaises(storage.ModelIncompatibleError):
            storage.load_model("sarsa", "other-signature", env.n_states, env.n_actions, self.models)
        with self.assertRaises(storage.ModelIncompatibleError):
            storage.load_model("sarsa", env.signature(), env.n_states + 1, env.n_actions, self.models)
        storage.model_path("sarsa", self.models).write_bytes(b"not a model")
        with self.assertRaises(storage.ModelIncompatibleError):
            storage.load_model("sarsa", env.signature(), env.n_states, env.n_actions, self.models)

    def test_retraining_one_algorithm_keeps_the_other(self):
        train_and_save("q_learning", small_settings(), seed=1,
                       models_dir=self.models, results_dir=self.results)
        before = storage.model_path("q_learning", self.models).read_bytes()
        train_and_save("sarsa", small_settings(), seed=2,
                       models_dir=self.models, results_dir=self.results)
        self.assertEqual(before, storage.model_path("q_learning", self.models).read_bytes())
        with open(self.results / "evaluation_summary.csv", newline="") as handle:
            names = {row["algorithm"] for row in csv.DictReader(handle)}
        self.assertEqual(names, {"Q-Learning", "SARSA"})

    def test_evaluation_metrics_are_valid(self):
        result = train("q_learning", small_settings(300), seed=1)
        metrics = evaluate_q_table(result.agent.q_table, SpaceEnvironment(), episodes=30)
        self.assertEqual(metrics["episodes"], 30)
        total = metrics["success_rate"] + metrics["collision_rate"] + metrics["timeout_rate"]
        self.assertAlmostEqual(total, 100.0)
        self.assertGreaterEqual(metrics["mean_crystals"], 0)

    def test_evaluation_without_success_reports_na(self):
        env = SpaceEnvironment()
        metrics = evaluate_q_table(np.zeros((env.n_states, env.n_actions)), env,
                                   episodes=10, noise=0.0)
        self.assertEqual(metrics["success_rate"], 0.0)
        self.assertIsNone(metrics["mean_steps_success"])
        self.assertEqual(format_steps(metrics["mean_steps_success"]), "N/A")

    def test_comparison_graphs_can_be_generated(self):
        env = SpaceEnvironment()
        data = {}
        for algorithm in config.ALGORITHM_KEYS:
            result = train(algorithm, small_settings(), seed=1)
            data[algorithm] = {"history": result.history,
                               "evaluation": evaluate_q_table(result.agent.q_table, env, episodes=20),
                               "episodes": result.episodes_completed}
        for which in ("all",) + plots.PANELS:
            raw, size = plots.figure_to_rgba(plots.build_figure(data, which, figsize=(6, 3.5), dpi=60))
            self.assertEqual(len(raw), size[0] * size[1] * 4)
        png = plots.save_comparison_png(data, self.results / "comparison.png")
        self.assertGreater(Path(png).stat().st_size, 1000)

    def test_graphs_handle_one_algorithm_and_no_successes(self):
        env = SpaceEnvironment()
        history = {key: np.zeros(20) for key in storage.HISTORY_KEYS}
        data = {"sarsa": {"history": history,
                          "evaluation": evaluate_q_table(np.zeros((env.n_states, 4)), env, episodes=5, noise=0.0),
                          "episodes": 20}}
        plots.figure_to_rgba(plots.build_figure(data, "all", figsize=(6, 3.5), dpi=60))
        plots.figure_to_rgba(plots.build_figure({}, "summary", figsize=(4, 2), dpi=60))

    def test_multi_seed_experiment_writes_csv(self):
        rows, summary = run_multi_seed_experiment(small_settings(30), [1, 2],
                                                  results_dir=self.results)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(summary), 2)
        self.assertTrue((self.results / "multi_seed_results.csv").exists())
        self.assertTrue((self.results / "multi_seed_summary.csv").exists())

    def test_rolling_helpers(self):
        np.testing.assert_allclose(plots.rolling_mean([1, 2, 3, 4], 2), [1, 1.5, 2.5, 3.5])
        out = plots.rolling_steps_per_success([10, 20, 30], [0, 1, 1], 2)
        self.assertTrue(np.isnan(out[0]))
        self.assertAlmostEqual(out[2], 25.0)

    def test_settings_validation_and_persistence(self):
        self.assertEqual(config.parse_field("episodes", "250"), 250)
        for name, bad in (("episodes", "abc"), ("episodes", "2.5"), ("alpha", "0"),
                          ("gamma", "1.5"), ("alpha", "nan")):
            with self.assertRaises(ValueError):
                config.parse_field(name, bad)
        self.assertEqual(config.Settings.from_dict({"episodes": "x", "alpha": 0.3}).alpha, 0.3)
        self.assertEqual(config.Settings.from_dict({"episodes": "x"}).episodes, 2500)
        self.assertTrue(config.Settings(epsilon_start=0.1, epsilon_min=0.5).problems())


if __name__ == "__main__":
    unittest.main()
