import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from environment import (DOWN, LEFT, RIGHT, UP, SpaceEnvironment,  # noqa: E402
                         STATUS_CRASHED, STATUS_SUCCESS, STATUS_TIMEOUT)

# Small map used by most tests (row 0 is the top):
#   . C G
#   . A .
#   S . .
SMALL = [".CG",
         ".A.",
         "S.."]


class EnvironmentTests(unittest.TestCase):
    def make(self, layout=SMALL, **kwargs):
        return SpaceEnvironment(layout=layout, **kwargs)

    def test_reset_returns_start_state(self):
        env = self.make()
        env.step(UP)
        state = env.reset()
        self.assertEqual(state, (0, 2, 0))
        self.assertEqual(env.steps, 0)
        self.assertEqual(env.total_reward, 0)
        self.assertFalse(env.done)

    def test_default_map_starts_bottom_left_and_goal_top_right(self):
        env = SpaceEnvironment()
        self.assertEqual((env.cols, env.rows), (10, 8))
        self.assertEqual(env.position, (0, 7))
        self.assertEqual(env.goal, (9, 0))
        self.assertEqual(env.n_crystals, 5)
        self.assertEqual(env.n_actions, 4)

    def test_all_four_actions(self):
        env = self.make()
        env.step(UP)
        self.assertEqual(env.position, (0, 1))
        env.step(RIGHT)
        self.assertEqual(env.position, (1, 1) if (1, 1) not in env.asteroids else env.position)
        env = self.make(layout=["...", "...", "S.G"])
        env.step(UP)
        self.assertEqual(env.position, (0, 1))
        env.step(RIGHT)
        self.assertEqual(env.position, (1, 1))
        env.step(DOWN)
        self.assertEqual(env.position, (1, 2))
        env.step(LEFT)
        self.assertEqual(env.position, (0, 2))

    def test_cannot_leave_grid_and_pays_step_penalty(self):
        env = self.make()
        for action in (LEFT, DOWN):
            _, reward, done, info = env.step(action)
            self.assertEqual(env.position, (0, 2))
            self.assertEqual(reward, config.REWARDS["step"])
            self.assertTrue(info["hit_boundary"])
            self.assertFalse(done)

    def test_crystal_rewarded_once_per_episode(self):
        env = self.make(layout=["C.G", "...", "S.."])
        env.step(UP)
        _, reward, _, info = env.step(UP)
        self.assertEqual(env.position, (0, 0))
        self.assertEqual(reward, config.REWARDS["crystal"])
        self.assertTrue(info["crystal_collected"])
        env.step(DOWN)
        _, reward, _, info = env.step(UP)  # back onto the same crystal cell
        self.assertEqual(reward, config.REWARDS["step"])
        self.assertFalse(info["crystal_collected"])
        self.assertEqual(env.crystals_collected, 1)
        # A new episode makes the crystal available again.
        env.reset()
        env.step(UP)
        _, reward, _, _ = env.step(UP)
        self.assertEqual(reward, config.REWARDS["crystal"])

    def test_crystal_changes_state(self):
        env = self.make(layout=["C.G", "...", "S.."])
        before = env.state_to_index()
        env.step(UP)
        state, _, _, _ = env.step(UP)
        self.assertEqual(state[2], 1)
        self.assertNotEqual(env.state_to_index(state), before)

    def test_asteroid_ends_episode(self):
        env = self.make()
        env.step(RIGHT)
        _, reward, done, info = env.step(UP)
        self.assertEqual(env.position, (1, 1))
        self.assertTrue(done)
        self.assertEqual(env.status, STATUS_CRASHED)
        self.assertEqual(reward, config.REWARDS["asteroid"])
        self.assertTrue(info["terminated"])

    def test_goal_ends_episode_successfully(self):
        env = self.make(layout=["..G", "...", "S.."])
        for action in (RIGHT, RIGHT, UP):
            env.step(action)
        _, reward, done, info = env.step(UP)
        self.assertTrue(done)
        self.assertEqual(env.status, STATUS_SUCCESS)
        self.assertEqual(reward, config.REWARDS["goal"])
        self.assertTrue(info["terminated"])

    def test_step_limit_truncates(self):
        env = self.make(layout=["..G", "...", "S.."], max_steps=3)
        done = False
        for _ in range(3):
            _, _, done, info = env.step(LEFT)
        self.assertTrue(done)
        self.assertEqual(env.status, STATUS_TIMEOUT)
        self.assertTrue(info["truncated"])
        self.assertFalse(info["terminated"])

    def test_step_after_done_raises(self):
        env = self.make(layout=["..G", "...", "S.."], max_steps=1)
        env.step(LEFT)
        with self.assertRaises(RuntimeError):
            env.step(LEFT)

    def test_invalid_action_raises(self):
        env = self.make()
        for bad in (4, -1, "x", None):
            with self.assertRaises(ValueError):
                env.step(bad)

    def test_unsolvable_map_rejected(self):
        with self.assertRaises(ValueError):
            SpaceEnvironment(layout=["AG", "SA"])

    def test_default_map_is_solvable_and_state_indices_unique_in_range(self):
        env = SpaceEnvironment()
        indices = {env.state_to_index((x, y, m)) for x in range(env.cols)
                   for y in range(env.rows) for m in range(1 << env.n_crystals)}
        self.assertEqual(len(indices), env.n_states)
        self.assertEqual(min(indices), 0)
        self.assertEqual(max(indices), env.n_states - 1)

    def test_render_returns_text(self):
        self.assertIn("@", SpaceEnvironment().render())


if __name__ == "__main__":
    unittest.main()
