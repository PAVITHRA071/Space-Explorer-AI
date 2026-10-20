"""Training of the agents (no graphics code in here).

``Trainer`` plays episodes in small slices (``run_episodes``), so the Pygame
interface can train a little every frame and stay responsive; the same class
is used for command-line training.
"""

import statistics
import time
from dataclasses import dataclass

import numpy as np

import config
from agents import create_agent
from environment import SpaceEnvironment
from evaluation.evaluate_agents import evaluate_q_table
from utils import storage


@dataclass
class TrainingResult:
    algorithm: str
    agent: object
    history: dict          # name -> NumPy array (one value per episode)
    seed: int
    episodes_completed: int
    requested_episodes: int
    cancelled: bool
    duration: float


class Trainer:
    """Trains one algorithm on one environment."""

    def __init__(self, algorithm, settings, seed=None, env=None):
        self.algorithm = algorithm
        self.settings = settings
        self.seed = seed
        self.env = env if env is not None else SpaceEnvironment()
        self.agent = create_agent(algorithm, self.env.n_states, self.env.n_actions,
                                  settings.hyperparameters(), seed=seed)
        self.total_episodes = int(settings.episodes)
        self.history = {key: [] for key in storage.HISTORY_KEYS}
        self.cancelled = False
        self._seconds = 0.0

    # ------------------------------------------------------------------
    @property
    def episode(self):
        return len(self.history["reward"])

    @property
    def finished(self):
        return self.cancelled or self.episode >= self.total_episodes

    def stop(self):
        """Request a safe stop; the current episode is never cut in half."""
        self.cancelled = True

    def run_episodes(self, count=None, time_budget=None):
        """Train for ``count`` episodes and/or ``time_budget`` seconds.

        At least one episode is played per call (unless already finished).
        Returns the number of episodes played."""
        started = time.perf_counter()
        played = 0
        while not self.finished:
            self._play_one_episode()
            played += 1
            if count is not None and played >= count:
                break
            if time_budget is not None and time.perf_counter() - started >= time_budget:
                break
        self._seconds += time.perf_counter() - started
        return played

    def _play_one_episode(self):
        epsilon_used = self.agent.epsilon
        outcome = self.agent.run_episode(self.env)
        self.agent.decay_epsilon()
        self.history["reward"].append(outcome["reward"])
        self.history["steps"].append(outcome["steps"])
        self.history["success"].append(1 if outcome["success"] else 0)
        self.history["crashed"].append(1 if outcome["crashed"] else 0)
        self.history["crystals"].append(outcome["crystals"])
        self.history["epsilon"].append(epsilon_used)

    # ------------------------------------------------------------------
    def snapshot(self):
        """Numbers shown on the training screen."""
        window = config.RECENT_WINDOW
        recent_rewards = self.history["reward"][-window:]
        recent_success = self.history["success"][-window:]
        return {
            "algorithm": self.algorithm,
            "episode": self.episode,
            "total": self.total_episodes,
            "epsilon": self.agent.epsilon,
            "last_reward": self.history["reward"][-1] if self.episode else 0.0,
            "avg_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            "success_rate": 100.0 * float(np.mean(recent_success)) if recent_success else 0.0,
            "progress": self.episode / self.total_episodes if self.total_episodes else 1.0,
        }

    def result(self):
        return TrainingResult(
            algorithm=self.algorithm,
            agent=self.agent,
            history={key: np.asarray(values, dtype=np.float64)
                     for key, values in self.history.items()},
            seed=self.seed,
            episodes_completed=self.episode,
            requested_episodes=self.total_episodes,
            cancelled=self.cancelled,
            duration=self._seconds,
        )


def train(algorithm, settings, seed=None, progress_callback=None, should_stop=None, env=None):
    """Train to completion (or until ``should_stop()`` is true).

    ``progress_callback(snapshot)`` is called roughly every 50 episodes."""
    trainer = Trainer(algorithm, settings, seed=seed, env=env)
    while not trainer.finished:
        if should_stop is not None and should_stop():
            trainer.stop()
            break
        trainer.run_episodes(count=50)
        if progress_callback is not None:
            progress_callback(trainer.snapshot())
    return trainer.result()


def finalize_training(result, env, models_dir=None, results_dir=None):
    """Evaluate a finished (not cancelled) run and save everything.

    Saves the model, the metadata, the training history CSV and the row in
    the evaluation summary CSV.  Returns the evaluation metrics."""
    if result.cancelled:
        raise ValueError("A cancelled training run is not saved.")
    evaluation = evaluate_q_table(result.agent.q_table, env)
    storage.save_model(
        result.algorithm, result.agent.q_table, result.history,
        result.agent.hyperparameters(), result.seed, result.episodes_completed,
        env.signature(), evaluation, models_dir)
    storage.save_history_csv(result.algorithm, result.history, results_dir)
    storage.update_evaluation_summary(
        result.algorithm, evaluation, result.episodes_completed, result.seed, results_dir)
    return evaluation


def train_and_save(algorithm, settings, seed=None, progress_callback=None,
                   models_dir=None, results_dir=None):
    """Train, evaluate and save one algorithm.  Returns ``(result, evaluation)``."""
    env = SpaceEnvironment()
    result = train(algorithm, settings, seed=seed, progress_callback=progress_callback, env=env)
    evaluation = finalize_training(result, env, models_dir, results_dir)
    return result, evaluation


def run_multi_seed_experiment(settings, seeds, algorithms=config.ALGORITHM_KEYS,
                              progress_callback=None, results_dir=None):
    """Train every algorithm once per seed and compare the evaluations.

    Saved models are NOT touched.  Writes ``multi_seed_results.csv`` (one row
    per algorithm and seed) and ``multi_seed_summary.csv`` (mean and standard
    deviation over the seeds).  Returns ``(rows, summary_rows)``."""
    seeds = list(seeds)
    if not seeds:
        raise ValueError("Give at least one seed.")
    rows = []
    for seed in seeds:
        for algorithm in algorithms:
            env = SpaceEnvironment()
            result = train(algorithm, settings, seed=seed, env=env)
            ev = evaluate_q_table(result.agent.q_table, env)
            steps = ev["mean_steps_success"]
            rows.append({
                "algorithm": config.ALGORITHM_NAMES[algorithm],
                "seed": seed,
                "episodes": result.episodes_completed,
                "mean_reward": round(ev["mean_reward"], 3),
                "success_rate_percent": round(ev["success_rate"], 2),
                "mean_steps_success": "N/A" if steps is None else round(steps, 3),
                "collision_rate_percent": round(ev["collision_rate"], 2),
                "mean_crystals": round(ev["mean_crystals"], 3),
                "final_train_success_percent": round(
                    100.0 * float(np.mean(result.history["success"][-config.RECENT_WINDOW:])), 2),
            })
            if progress_callback is not None:
                progress_callback(algorithm, seed)

    summary_rows = []
    for algorithm in algorithms:
        name = config.ALGORITHM_NAMES[algorithm]
        mine = [row for row in rows if row["algorithm"] == name]

        def stats(field):
            values = [row[field] for row in mine if row[field] != "N/A"]
            if not values:
                return "N/A", "N/A"
            spread = statistics.pstdev(values) if len(values) > 1 else 0.0
            return round(statistics.fmean(values), 3), round(spread, 3)

        summary = {"algorithm": name, "seeds": len(mine), "episodes": settings.episodes}
        for field in ("mean_reward", "success_rate_percent", "mean_steps_success",
                      "collision_rate_percent", "mean_crystals"):
            mean, spread = stats(field)
            summary[f"{field}_mean"] = mean
            summary[f"{field}_std"] = spread
        summary_rows.append(summary)

    storage.save_multi_seed_csv(rows, summary_rows, results_dir)
    return rows, summary_rows
