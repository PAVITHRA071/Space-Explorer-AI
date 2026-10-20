"""The Space Explorer grid world (no graphics, no learning code).

The environment follows the familiar reinforcement-learning interface:

    state = env.reset()
    state, reward, done, info = env.step(action)

It knows nothing about Pygame, so it can be used for training, testing and
for the on-screen game alike.
"""

import hashlib
import json
from collections import deque

import config

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ("Up", "Down", "Left", "Right")
# (dx, dy) for each action.  y grows DOWNWARDS (row 0 is the top row).
ACTION_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
N_ACTIONS = 4

# Episode status values
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_CRASHED = "crashed"
STATUS_TIMEOUT = "timeout"


class SpaceEnvironment:
    """A grid with asteroids, energy crystals, a start cell and a destination.

    State
        ``(x, y, crystal_mask)`` where ``crystal_mask`` is a bit-field: bit *i*
        is 1 once crystal *i* has been collected.  Because the mask is part of
        the state, the agent can learn that a crystal is only worth visiting
        until it has been collected.

    Rewards (see ``config.REWARDS``)
        ordinary move or bump into the boundary ... -1
        collecting a NEW crystal ................. +20 (replaces the -1)
        reaching the destination ................. +100 (replaces the -1)
        hitting an asteroid ...................... -100 (replaces the -1)

    An episode ends when the ship reaches the destination, hits an asteroid
    or has made ``max_steps`` moves.
    """

    def __init__(self, layout=None, max_steps=None, rewards=None):
        self.layout = list(layout if layout is not None else config.DEFAULT_LAYOUT)
        self.max_steps = int(max_steps if max_steps is not None else config.MAX_STEPS)
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        self.rewards = dict(config.REWARDS)
        if rewards:
            self.rewards.update(rewards)

        self._parse_layout()
        self._check_solvable()

        self.n_actions = N_ACTIONS
        self.n_states = self.cols * self.rows * (1 << len(self.crystals))

        # Episode variables (initialised properly by reset()).
        self.position = self.start
        self.crystal_mask = 0
        self.steps = 0
        self.total_reward = 0
        self.status = STATUS_RUNNING
        self.done = False
        self.reset()

    # ------------------------------------------------------------------
    # Map construction
    # ------------------------------------------------------------------
    def _parse_layout(self):
        if not self.layout:
            raise ValueError("The map layout is empty.")
        widths = {len(row) for row in self.layout}
        if len(widths) != 1:
            raise ValueError("All map rows must have the same length.")
        self.rows = len(self.layout)
        self.cols = widths.pop()

        start = goal = None
        asteroids, crystals = set(), []
        for y, row in enumerate(self.layout):
            for x, symbol in enumerate(row):
                if symbol == "S":
                    if start is not None:
                        raise ValueError("The map must contain exactly one start 'S'.")
                    start = (x, y)
                elif symbol == "G":
                    if goal is not None:
                        raise ValueError("The map must contain exactly one destination 'G'.")
                    goal = (x, y)
                elif symbol == "A":
                    asteroids.add((x, y))
                elif symbol == "C":
                    crystals.append((x, y))  # row-major order => stable bit numbers
                elif symbol != ".":
                    raise ValueError(f"Unknown map symbol {symbol!r}; use . A C S G")
        if start is None or goal is None:
            raise ValueError("The map needs one start 'S' and one destination 'G'.")

        self.start = start
        self.goal = goal
        self.asteroids = frozenset(asteroids)
        self.crystals = tuple(crystals)
        self._crystal_bit = {pos: 1 << i for i, pos in enumerate(self.crystals)}

    def _check_solvable(self):
        """Breadth-first search over safe cells: the goal and every crystal
        must be reachable from the start without touching an asteroid."""
        seen = {self.start}
        queue = deque([self.start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ACTION_DELTAS:
                nxt = (x + dx, y + dy)
                if (0 <= nxt[0] < self.cols and 0 <= nxt[1] < self.rows
                        and nxt not in self.asteroids and nxt not in seen):
                    seen.add(nxt)
                    queue.append(nxt)
        if self.goal not in seen:
            raise ValueError("Unsolvable map: the destination cannot be reached safely.")
        unreachable = [c for c in self.crystals if c not in seen]
        if unreachable:
            raise ValueError(f"Unsolvable map: crystals {unreachable} cannot be reached safely.")

    # ------------------------------------------------------------------
    # Core reinforcement-learning interface
    # ------------------------------------------------------------------
    def reset(self):
        """Start a new mission and return the initial state."""
        self.position = self.start
        self.crystal_mask = 0
        self.steps = 0
        self.total_reward = 0
        self.status = STATUS_RUNNING
        self.done = False
        return self.get_state()

    def get_state(self):
        """Return ``(x, y, crystal_mask)``."""
        return (self.position[0], self.position[1], self.crystal_mask)

    def state_to_index(self, state=None):
        """Convert a state tuple into a row number of the Q-table."""
        x, y, mask = state if state is not None else self.get_state()
        return mask * (self.cols * self.rows) + y * self.cols + x

    def step(self, action):
        """Apply ``action`` and return ``(state, reward, done, info)``."""
        if self.done:
            raise RuntimeError("The mission is over; call reset() to start a new one.")
        try:
            action = int(action)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid action {action!r}; use 0, 1, 2 or 3.") from None
        if action not in (UP, DOWN, LEFT, RIGHT):
            raise ValueError(f"Invalid action {action!r}; use 0, 1, 2 or 3.")

        x, y = self.position
        dx, dy = ACTION_DELTAS[action]
        new_x, new_y = x + dx, y + dy

        # Moves outside the grid are ignored: the ship stays where it is.
        hit_boundary = not (0 <= new_x < self.cols and 0 <= new_y < self.rows)
        if hit_boundary:
            new_x, new_y = x, y

        self.position = (new_x, new_y)
        self.steps += 1

        reward = self.rewards["step"]
        terminated = False
        crystal_collected = False

        if self.position in self.asteroids:
            reward = self.rewards["asteroid"]
            terminated = True
            self.status = STATUS_CRASHED
        elif self.position == self.goal:
            reward = self.rewards["goal"]
            terminated = True
            self.status = STATUS_SUCCESS
        else:
            bit = self._crystal_bit.get(self.position)
            # A crystal pays out only the first time it is collected.
            if bit is not None and not (self.crystal_mask & bit):
                self.crystal_mask |= bit
                reward = self.rewards["crystal"]
                crystal_collected = True

        # Running out of fuel ends the episode without a terminal event.
        truncated = (not terminated) and self.steps >= self.max_steps
        if truncated:
            self.status = STATUS_TIMEOUT

        self.done = terminated or truncated
        self.total_reward += reward

        info = {
            "terminated": terminated,          # reached goal or crashed
            "truncated": truncated,            # step limit reached
            "hit_boundary": hit_boundary,
            "crystal_collected": crystal_collected,
            "crystals_collected": self.crystals_collected,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "status": self.status,
            "position": self.position,
        }
        return self.get_state(), reward, self.done, info

    # ------------------------------------------------------------------
    # Helpers used by the interface, evaluation and tests
    # ------------------------------------------------------------------
    @property
    def crystals_collected(self):
        return bin(self.crystal_mask).count("1")

    @property
    def n_crystals(self):
        return len(self.crystals)

    def remaining_crystals(self):
        """Positions of crystals that have not been collected yet."""
        return [pos for pos, bit in self._crystal_bit.items() if not (self.crystal_mask & bit)]

    def signature(self):
        """Short fingerprint of the map, rewards and step limit.

        Saved with every model so a model trained for a different setup is
        never loaded by mistake."""
        description = json.dumps(
            {"layout": self.layout, "rewards": self.rewards, "max_steps": self.max_steps},
            sort_keys=True,
        )
        return hashlib.sha256(description.encode("utf-8")).hexdigest()[:16]

    def render(self):
        """Return a text drawing of the current state.

        @ ship   A asteroid   C crystal   P planet (destination)   . empty
        """
        remaining = set(self.remaining_crystals())
        lines = []
        for y in range(self.rows):
            chars = []
            for x in range(self.cols):
                pos = (x, y)
                if pos == self.position:
                    chars.append("@")
                elif pos in self.asteroids:
                    chars.append("A")
                elif pos == self.goal:
                    chars.append("P")
                elif pos in remaining:
                    chars.append("C")
                else:
                    chars.append(".")
            lines.append(" ".join(chars))
        return "\n".join(lines)
