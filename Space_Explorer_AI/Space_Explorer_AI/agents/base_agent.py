"""Parts shared by the Q-Learning and SARSA agents.

Only the things that are *identical* in both algorithms live here: the
Q-table, epsilon-greedy action selection and the exploration decay.  The
learning rule itself (the difference between the two algorithms) is written
separately in ``q_learning.py`` and ``sarsa.py``.
"""

import numpy as np


class TabularAgent:
    """An agent that stores one value ``Q[state, action]`` in a table."""

    name = "Tabular agent"
    key = "tabular"

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_decay=0.998, epsilon_min=0.01, seed=None):
        if n_states < 1 or n_actions < 1:
            raise ValueError("n_states and n_actions must be positive.")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= epsilon_min <= epsilon_start <= 1.0:
            raise ValueError("Need 0 <= epsilon_min <= epsilon_start <= 1.")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")

        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha                # learning rate
        self.gamma = gamma                # discount factor
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start      # current exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Q-table: one row per state, one column per action, all zeros at first.
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def choose_action(self, state_index):
        """Epsilon-greedy: explore with probability epsilon, otherwise exploit.

        Ties between equally good actions are broken randomly so the agent
        does not always favour "Up" while the table is still all zeros."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.q_table[state_index]
        best_actions = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best_actions))

    def greedy_action(self, state_index):
        """Pure exploitation: the action with the highest Q-value.

        ``np.argmax`` returns the FIRST maximum, so ties are always broken the
        same way (lowest action number: Up, Down, Left, Right)."""
        return int(np.argmax(self.q_table[state_index]))

    def decay_epsilon(self):
        """Reduce exploration a little after every episode (never below the minimum)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def hyperparameters(self):
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "final_epsilon": self.epsilon,
        }

    def set_q_table(self, q_table):
        """Install a previously saved Q-table (shape must match)."""
        q_table = np.asarray(q_table, dtype=np.float64)
        if q_table.shape != (self.n_states, self.n_actions):
            raise ValueError(
                f"Q-table shape {q_table.shape} does not match "
                f"({self.n_states}, {self.n_actions})."
            )
        self.q_table = q_table.copy()


def episode_result(env, total_reward=None):
    """Summarise a finished episode as a plain dictionary."""
    return {
        "reward": env.total_reward if total_reward is None else total_reward,
        "steps": env.steps,
        "success": env.status == "success",
        "crashed": env.status == "crashed",
        "timed_out": env.status == "timeout",
        "crystals": env.crystals_collected,
    }
