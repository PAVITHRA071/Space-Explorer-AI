"""Evaluation of trained agents (separate from training).

During training the agent explores randomly, which distorts its results.  So
after training we run a separate EVALUATION phase with a fixed policy:

* the agent always takes the greedy action (highest Q-value, ties broken by
  the lowest action number),
* with a small fixed probability (``EVAL_ACTION_NOISE``) a random move is made
  instead, imitating engine glitches.  Without this the map is deterministic,
  all evaluation episodes would be identical and "200 episodes" would be
  meaningless.  The noise also reveals which policy keeps a safe distance
  from asteroids.
* a fixed seed gives both algorithms exactly the same random generator.

Run ``python -m evaluation.evaluate_agents`` to evaluate the saved models.
"""

import sys

import numpy as np

import config
from environment import SpaceEnvironment
from utils import storage


def evaluate_q_table(q_table, env, episodes=None, noise=None, seed=None):
    """Evaluate a Q-table and return a dictionary of metrics.

    Percentages are in the range 0-100.  ``mean_steps_success`` is ``None``
    when no episode succeeded (shown as "N/A")."""
    episodes = config.EVAL_EPISODES if episodes is None else int(episodes)
    noise = config.EVAL_ACTION_NOISE if noise is None else float(noise)
    seed = config.EVAL_SEED if seed is None else seed
    if episodes < 1:
        raise ValueError("Need at least one evaluation episode.")
    if not 0.0 <= noise <= 1.0:
        raise ValueError("Evaluation noise must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    rewards, crystals = [], []
    success_steps = []
    crashes = timeouts = 0
    for _ in range(episodes):
        state = env.state_to_index(env.reset())
        done = False
        while not done:
            if noise > 0 and rng.random() < noise:
                action = int(rng.integers(env.n_actions))
            else:
                action = int(np.argmax(q_table[state]))
            next_state, _, done, _ = env.step(action)
            state = env.state_to_index(next_state)
        rewards.append(env.total_reward)
        crystals.append(env.crystals_collected)
        if env.status == "success":
            success_steps.append(env.steps)
        elif env.status == "crashed":
            crashes += 1
        else:
            timeouts += 1

    return {
        "episodes": episodes,
        "noise": noise,
        "mean_reward": float(np.mean(rewards)),
        "success_rate": 100.0 * len(success_steps) / episodes,
        "mean_steps_success": float(np.mean(success_steps)) if success_steps else None,
        "collision_rate": 100.0 * crashes / episodes,
        "timeout_rate": 100.0 * timeouts / episodes,
        "mean_crystals": float(np.mean(crystals)),
    }


def format_steps(value):
    """Show 'N/A' instead of a number when there were no successful episodes."""
    return "N/A" if value is None else f"{value:.1f}"


def evaluate_saved_models(models_dir=None, results_dir=None):
    """Evaluate every saved model and update ``results/evaluation_summary.csv``.

    Returns ``{algorithm: metrics-or-error-message}``."""
    env = SpaceEnvironment()
    outcome = {}
    for algorithm in config.ALGORITHM_KEYS:
        try:
            model = storage.load_model(algorithm, env.signature(), env.n_states,
                                       env.n_actions, models_dir)
        except storage.ModelError as error:
            outcome[algorithm] = str(error)
            continue
        metrics = evaluate_q_table(model.q_table, env)
        storage.update_evaluation_summary(
            algorithm, metrics, model.meta.get("episodes", 0), model.meta.get("seed"),
            results_dir)
        outcome[algorithm] = metrics
    return outcome


def main():
    outcome = evaluate_saved_models()
    for algorithm, result in outcome.items():
        name = config.ALGORITHM_NAMES[algorithm]
        if isinstance(result, str):
            print(f"{name}: {result}")
            continue
        print(f"{name}: mean reward {result['mean_reward']:.1f}, "
              f"success {result['success_rate']:.1f}%, "
              f"steps/success {format_steps(result['mean_steps_success'])}, "
              f"collisions {result['collision_rate']:.1f}%, "
              f"crystals {result['mean_crystals']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
