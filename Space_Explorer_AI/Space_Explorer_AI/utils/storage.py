"""Saving and loading of models, training history, results and settings.

Design choices
* Models are stored as NumPy ``.npz`` files that are loaded with
  ``allow_pickle=False`` plus a small JSON file with the metadata.  Unlike
  pickle, these formats cannot run code, so a model file from somebody else
  can never execute anything on your computer.  (You should still only use
  model files that you trained yourself.)
* Every algorithm has its OWN files, so retraining one algorithm can never
  overwrite the other one's model.
* Files are written to a temporary name first and then renamed, so a crash
  while saving cannot leave a half-written model behind.
* All paths are relative to the project folder (see ``config.py``).
"""

import csv
import json
import os
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

import config

HISTORY_KEYS = ("reward", "steps", "success", "crashed", "crystals", "epsilon")
EVALUATION_FIELDS = (
    "algorithm", "episodes_trained", "eval_episodes", "eval_action_noise",
    "mean_reward", "success_rate_percent", "mean_steps_success",
    "collision_rate_percent", "timeout_rate_percent", "mean_crystals",
    "seed", "saved_at",
)

LoadedModel = namedtuple("LoadedModel", ["q_table", "history", "meta"])


class ModelError(Exception):
    """Base class for model loading problems (message is safe to show users)."""


class ModelNotFoundError(ModelError):
    """No saved model exists for the requested algorithm."""


class ModelIncompatibleError(ModelError):
    """The saved model is damaged or was trained for a different setup."""


def _models_dir(models_dir):
    return Path(models_dir) if models_dir is not None else config.MODELS_DIR


def _results_dir(results_dir):
    return Path(results_dir) if results_dir is not None else config.RESULTS_DIR


def model_path(algorithm, models_dir=None):
    return _models_dir(models_dir) / f"{algorithm}_model.npz"


def meta_path(algorithm, models_dir=None):
    return _models_dir(models_dir) / f"{algorithm}_meta.json"


def _json_default(value):
    """Let json.dump handle NumPy numbers."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot save {type(value).__name__} as JSON.")


def _atomic_write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def save_model(algorithm, q_table, history, hyperparameters, seed, episodes,
               signature, evaluation=None, models_dir=None):
    """Save one algorithm's Q-table, training history and metadata."""
    directory = _models_dir(models_dir)
    directory.mkdir(parents=True, exist_ok=True)

    arrays = {"q_table": np.asarray(q_table, dtype=np.float64)}
    for key in HISTORY_KEYS:
        arrays[f"h_{key}"] = np.asarray(history[key], dtype=np.float64)

    npz_target = model_path(algorithm, models_dir)
    npz_tmp = npz_target.with_name(npz_target.name + ".tmp")
    with open(npz_tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(npz_tmp, npz_target)

    meta = {
        "algorithm": algorithm,
        "algorithm_name": config.ALGORITHM_NAMES.get(algorithm, algorithm),
        "episodes": int(episodes),
        "seed": None if seed is None else int(seed),
        "hyperparameters": hyperparameters,
        "environment_signature": signature,
        "q_table_shape": list(arrays["q_table"].shape),
        "evaluation": evaluation,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _atomic_write_text(meta_path(algorithm, models_dir),
                       json.dumps(meta, indent=2, default=_json_default))
    return meta


def load_model(algorithm, signature, n_states, n_actions, models_dir=None):
    """Load a saved model after checking that it matches the current game.

    Raises ``ModelNotFoundError`` or ``ModelIncompatibleError`` with a message
    that explains what the user should do.
    """
    name = config.ALGORITHM_NAMES.get(algorithm, algorithm)
    npz_file = model_path(algorithm, models_dir)
    json_file = meta_path(algorithm, models_dir)
    if not npz_file.exists() or not json_file.exists():
        raise ModelNotFoundError(
            f"No saved {name} model found. Train it first from the main menu.")

    try:
        with open(json_file, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        with np.load(npz_file, allow_pickle=False) as data:
            q_table = np.array(data["q_table"], dtype=np.float64)
            history = {key: np.array(data[f"h_{key}"], dtype=np.float64)
                       for key in HISTORY_KEYS}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ModelIncompatibleError(
            f"The saved {name} model is damaged ({error}). Please train it again.") from error

    if not isinstance(meta, dict) or meta.get("environment_signature") != signature:
        raise ModelIncompatibleError(
            f"The saved {name} model was trained for a different map or reward setup. "
            "Please train it again.")
    if q_table.shape != (n_states, n_actions) or not np.all(np.isfinite(q_table)):
        raise ModelIncompatibleError(
            f"The saved {name} model does not match the game state size. "
            "Please train it again.")
    return LoadedModel(q_table, history, meta)


def model_info(algorithm, models_dir=None):
    """Return the metadata of a saved model, None if there is none, or a
    dictionary ``{"error": message}`` if the metadata file is unreadable."""
    json_file = meta_path(algorithm, models_dir)
    if not json_file.exists() or not model_path(algorithm, models_dir).exists():
        return None
    try:
        with open(json_file, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if not isinstance(meta, dict):
            raise ValueError("unexpected file content")
        return meta
    except (OSError, ValueError) as error:
        return {"error": str(error)}


# --------------------------------------------------------------------------
# CSV results (for the college report)
# --------------------------------------------------------------------------
def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def save_history_csv(algorithm, history, results_dir=None):
    """Write one row per training episode to ``training_history_<algorithm>.csv``."""
    path = _results_dir(results_dir) / f"training_history_{algorithm}.csv"
    count = len(history["reward"])
    rows = []
    for i in range(count):
        rows.append({
            "episode": i + 1,
            "reward": float(history["reward"][i]),
            "steps": int(history["steps"][i]),
            "success": int(history["success"][i]),
            "crashed": int(history["crashed"][i]),
            "crystals": int(history["crystals"][i]),
            "epsilon": round(float(history["epsilon"][i]), 6),
        })
    _write_csv(path, ("episode",) + HISTORY_KEYS, rows)
    return path


def _summary_row(algorithm, evaluation, episodes_trained, seed):
    steps = evaluation.get("mean_steps_success")
    return {
        "algorithm": config.ALGORITHM_NAMES.get(algorithm, algorithm),
        "episodes_trained": int(episodes_trained),
        "eval_episodes": evaluation["episodes"],
        "eval_action_noise": evaluation["noise"],
        "mean_reward": round(evaluation["mean_reward"], 3),
        "success_rate_percent": round(evaluation["success_rate"], 2),
        "mean_steps_success": "N/A" if steps is None else round(steps, 3),
        "collision_rate_percent": round(evaluation["collision_rate"], 2),
        "timeout_rate_percent": round(evaluation["timeout_rate"], 2),
        "mean_crystals": round(evaluation["mean_crystals"], 3),
        "seed": "" if seed is None else seed,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def update_evaluation_summary(algorithm, evaluation, episodes_trained, seed, results_dir=None):
    """Insert or replace this algorithm's row in ``evaluation_summary.csv``.

    The other algorithm's row is kept untouched."""
    path = _results_dir(results_dir) / "evaluation_summary.csv"
    rows = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows[row["algorithm"]] = row
        except (OSError, KeyError, csv.Error):
            rows = {}  # unreadable old file: it is rebuilt from scratch
    new_row = _summary_row(algorithm, evaluation, episodes_trained, seed)
    rows[new_row["algorithm"]] = new_row
    ordered = [rows[name] for name in sorted(rows)]
    _write_csv(path, EVALUATION_FIELDS, ordered)
    return path


def save_multi_seed_csv(rows, summary_rows, results_dir=None):
    """Write the per-seed table and the mean/std summary of a multi-seed run."""
    directory = _results_dir(results_dir)
    per_seed = directory / "multi_seed_results.csv"
    summary = directory / "multi_seed_summary.csv"
    if rows:
        _write_csv(per_seed, list(rows[0].keys()), rows)
    if summary_rows:
        _write_csv(summary, list(summary_rows[0].keys()), summary_rows)
    return per_seed, summary


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
def load_settings():
    """Load ``settings.json``; use the defaults if it is missing or invalid."""
    try:
        with open(config.SETTINGS_PATH, "r", encoding="utf-8") as handle:
            return config.Settings.from_dict(json.load(handle))
    except (OSError, ValueError):
        return config.Settings()


def save_settings(settings):
    """Save the settings (raises ``OSError`` if the file cannot be written)."""
    _atomic_write_text(config.SETTINGS_PATH, json.dumps(settings.to_dict(), indent=2))
