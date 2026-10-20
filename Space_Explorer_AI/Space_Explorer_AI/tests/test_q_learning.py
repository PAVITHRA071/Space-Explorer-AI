import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import QLearningAgent  # noqa: E402
from environment import SpaceEnvironment  # noqa: E402


def make_agent(**kwargs):
    params = dict(n_states=5, n_actions=4, alpha=0.5, gamma=0.9, seed=0)
    params.update(kwargs)
    return QLearningAgent(**params)


class QLearningTests(unittest.TestCase):
    def test_table_starts_at_zero(self):
        self.assertTrue(np.all(make_agent().q_table == 0))

    def test_update_uses_max_of_next_state(self):
        agent = make_agent()
        agent.q_table[1] = [1.0, 5.0, 2.0, 0.0]   # max is 5
        agent.update(state=0, action=2, reward=-1, next_state=1, terminal=False)
        # 0 + 0.5 * (-1 + 0.9 * 5 - 0) = 1.75
        self.assertAlmostEqual(agent.q_table[0, 2], 1.75)

    def test_terminal_update_does_not_bootstrap(self):
        agent = make_agent()
        agent.q_table[1] = [100.0] * 4
        agent.update(0, 0, 10, 1, terminal=True)
        self.assertAlmostEqual(agent.q_table[0, 0], 5.0)  # 0.5 * 10

    def test_epsilon_decay_respects_minimum(self):
        agent = make_agent(epsilon_start=1.0, epsilon_decay=0.5, epsilon_min=0.2)
        for _ in range(10):
            agent.decay_epsilon()
        self.assertAlmostEqual(agent.epsilon, 0.2)

    def test_greedy_action_and_tie_break(self):
        agent = make_agent()
        self.assertEqual(agent.greedy_action(0), 0)  # all equal -> lowest index
        agent.q_table[0] = [0, 0, 3, 3]
        self.assertEqual(agent.greedy_action(0), 2)

    def test_epsilon_zero_is_greedy_and_one_explores(self):
        agent = make_agent(epsilon_start=0.0, epsilon_min=0.0)
        agent.q_table[0] = [0, 0, 0, 9]
        self.assertTrue(all(agent.choose_action(0) == 3 for _ in range(20)))
        agent = make_agent(epsilon_start=1.0)
        agent.q_table[0] = [0, 0, 0, 9]
        self.assertGreater(len({agent.choose_action(0) for _ in range(100)}), 1)

    def test_run_episode_changes_q_table_and_reports_stats(self):
        env = SpaceEnvironment()
        agent = QLearningAgent(env.n_states, env.n_actions, seed=1)
        stats = agent.run_episode(env)
        self.assertTrue(np.any(agent.q_table != 0))
        for key in ("reward", "steps", "success", "crashed", "timed_out", "crystals"):
            self.assertIn(key, stats)

    def test_learns_to_reach_goal_on_small_map(self):
        env = SpaceEnvironment(layout=["..G", ".A.", "S.."], max_steps=30)
        agent = QLearningAgent(env.n_states, env.n_actions, alpha=0.2, gamma=0.95,
                               epsilon_decay=0.99, seed=3)
        for _ in range(600):
            agent.run_episode(env)
            agent.decay_epsilon()
        state = env.state_to_index(env.reset())
        done = False
        while not done:
            state_tuple, _, done, _ = env.step(agent.greedy_action(state))
            state = env.state_to_index(state_tuple)
        self.assertEqual(env.status, "success")

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            make_agent(alpha=0)
        with self.assertRaises(ValueError):
            make_agent(gamma=1.5)


if __name__ == "__main__":
    unittest.main()
