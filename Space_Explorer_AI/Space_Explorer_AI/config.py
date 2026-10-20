"""Central configuration for Space Explorer AI.

Everything that a student might want to tweak lives here: the map, the reward
values, default hyper-parameters, evaluation settings and the user-interface
colours.  The other modules import these values instead of hard-coding them.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (always relative to the project folder, never to the current directory)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SETTINGS_PATH = PROJECT_ROOT / "settings.json"

# --------------------------------------------------------------------------
# Algorithms
# --------------------------------------------------------------------------
Q_LEARNING = "q_learning"
SARSA = "sarsa"
ALGORITHM_NAMES = {Q_LEARNING: "Q-Learning", SARSA: "SARSA"}
ALGORITHM_KEYS = (Q_LEARNING, SARSA)

# --------------------------------------------------------------------------
# Environment: map, rewards and episode length
# --------------------------------------------------------------------------
GRID_COLS = 10
GRID_ROWS = 8
MAX_STEPS = 100  # an episode ends after this many moves ("out of fuel")

# Map legend:  .  empty space      A  asteroid (deadly)
#              C  energy crystal   S  start      G  destination planet
# Row 0 is the TOP of the screen, so S (bottom-left) is the last row, first
# column and G (top-right) is the first row, last column.
DEFAULT_LAYOUT = [
    "..C...A..G",
    ".A..A...C.",
    "...A..C.A.",
    "A....A....",
    "..C....A..",
    ".A..A.....",
    ".....A.C..",
    "S..A......",
]

REWARDS = {
    "step": -1,        # ordinary move (also bumping into the boundary)
    "crystal": 20,     # collecting a NEW crystal (replaces the step penalty)
    "goal": 100,       # reaching the destination planet
    "asteroid": -100,  # flying into an asteroid
}

# --------------------------------------------------------------------------
# Training / evaluation defaults
# --------------------------------------------------------------------------
RECENT_WINDOW = 100      # window for "recent" averages shown during training
ROLLING_WINDOW = 100     # window for the smoothed comparison graphs
EVAL_EPISODES = 200      # number of evaluation episodes per algorithm
EVAL_ACTION_NOISE = 0.05  # probability of a random move during evaluation
EVAL_SEED = 12345        # fixed seed so both algorithms face the same noise

# Hyper-parameter / setting limits used for input validation.
# name: (label, type, minimum, maximum)
FIELD_SPECS = {
    "episodes": ("Training episodes", int, 10, 200000),
    "alpha": ("Learning rate (alpha)", float, 0.001, 1.0),
    "gamma": ("Discount factor (gamma)", float, 0.0, 0.999),
    "epsilon_start": ("Initial exploration (epsilon)", float, 0.0, 1.0),
    "epsilon_decay": ("Exploration decay per episode", float, 0.5, 1.0),
    "epsilon_min": ("Minimum exploration", float, 0.0, 1.0),
    "playback_speed": ("Playback speed (steps/second)", int, 1, 30),
    "seed": ("Random seed (-1 = random)", int, -1, 2**31 - 1),
}


def parse_field(name, value):
    """Convert ``value`` (text or number) into the right type for ``name``.

    Raises ``ValueError`` with a friendly message when the value is invalid.
    """
    label, kind, low, high = FIELD_SPECS[name]
    try:
        if kind is int:
            if isinstance(value, float):
                if not value.is_integer():
                    raise ValueError
                parsed = int(value)
            else:
                parsed = int(str(value).strip())
        else:
            parsed = float(str(value).strip())
    except (TypeError, ValueError):
        kind_word = "a whole number" if kind is int else "a number"
        raise ValueError(f"{label} must be {kind_word}.") from None
    if isinstance(parsed, float) and not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number.")
    if parsed < low or parsed > high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return parsed


@dataclass
class Settings:
    """User-adjustable settings (also the default hyper-parameters)."""

    episodes: int = 2500
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.998
    epsilon_min: float = 0.01
    playback_speed: int = 4
    seed: int = 42

    def to_dict(self):
        return asdict(self)

    def problems(self):
        """Return a list of cross-field problems (empty when consistent)."""
        problems = []
        if self.epsilon_min > self.epsilon_start:
            problems.append("Minimum exploration cannot exceed the initial exploration rate.")
        return problems

    def hyperparameters(self):
        """Only the values that an agent needs."""
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
        }

    @classmethod
    def from_dict(cls, data):
        """Build settings from stored data, falling back to defaults when needed."""
        settings = cls()
        if not isinstance(data, dict):
            return settings
        for name in FIELD_SPECS:
            if name in data:
                try:
                    setattr(settings, name, parse_field(name, data[name]))
                except ValueError:
                    pass  # keep the default for this field
        if settings.problems():
            return cls()
        return settings


# --------------------------------------------------------------------------
# User interface constants
# --------------------------------------------------------------------------
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 660  # small enough for 1366x768 laptop screens
FPS = 60
CELL_SIZE = 64
BOARD_ORIGIN = (24, 100)
PANEL_RECT = (688, 100, 312, 512)  # x, y, width, height of the status panel

COLORS = {
    "bg": (7, 11, 30),
    "bg_light": (12, 19, 48),
    "panel": (16, 24, 56),
    "panel_border": (46, 66, 130),
    "cell": (13, 21, 50),
    "cell_alt": (16, 26, 60),
    "grid_line": (34, 52, 110),
    "text": (232, 238, 255),
    "muted": (140, 156, 205),
    "cyan": (0, 212, 255),
    "blue": (70, 130, 255),
    "purple": (150, 92, 255),
    "yellow": (255, 213, 74),
    "green": (84, 240, 160),
    "red": (255, 86, 86),
    "orange": (255, 142, 52),
    "white": (255, 255, 255),
}

# Colours used for each algorithm in graphs and the interface.
ALGORITHM_COLORS = {Q_LEARNING: "#00d4ff", SARSA: "#ffc857"}
