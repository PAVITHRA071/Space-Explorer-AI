import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import QLearningAgent, SarsaAgent  # noqa: E402
from environment import SpaceEnvironment  # noqa: E402


def make_agent(**kwargs):
    params = dict(n_states=5, n_actions=4, alpha=0.5, gamma=0.9, seed=0)
    params.update(kwargs)
    return SarsaAgent(**params)


class SarsaTests(unittest.TestCase):
    def test_update_uses_selected_next_action_not_max(self):
        agent = make_agent()
        agent.q_table[1] = [1.0, 5.0, 2.0, 0.0]
        agent.update(0, 2, -1, 1, next_action=3, terminal=False)   # Q(s',a') = 0
        # 0 + 0.5 * (-1 + 0.9 * 0 - 0) = -0.5   (Q-Learning would give 1.75)
        self.assertAlmostEqual(agent.q_table[0, 2], -0.5)

    def test_sarsa_and_q_learning_targets_differ(self):
        sarsa, q_learning = make_agent(), QLearningAgent(5, 4, alpha=0.5, gamma=0.9, seed=0)
        for agent in (sarsa, q_learning):
            agent.q_table[1] = [1.0, 5.0, 2.0, 0.0]
        sarsa.update(0, 0, -1, 1, 2, False)
        q_learning.update(0, 0, -1, 1, False)
        self.assertNotAlmostEqual(sarsa.q_table[0, 0], q_learning.q_table[0, 0])

    def test_terminal_update_does_not_bootstrap(self):
        agent = make_agent()
        agent.q_table[1] = [100.0] * 4
        agent.update(0, 0, -100, 1, None, terminal=True)
        self.assertAlmostEqual(agent.q_table[0, 0], -50.0)

    def test_run_episode_uses_the_action_it_selected_for_the_update(self):
        """Wrap choose_action to record every action; each update must use the
        action that is then really taken."""
        env = SpaceEnvironment(layout=["..G", "...", "S.."], max_steps=6)
        agent = SarsaAgent(env.n_states, env.n_actions, epsilon_start=1.0, seed=5)
        chosen, updates = [], []
        original_choose, original_update = agent.choose_action, agent.update

        def choose(state):
            action = original_choose(state)
            chosen.append(action)
            return action

        def update(state, action, reward, next_state, next_action, terminal):
            updates.append((action, next_action, terminal))
            return original_update(state, action, reward, next_state, next_action, terminal)

        agent.choose_action, agent.update = choose, update
        agent.run_episode(env)
        self.assertGreater(len(updates), 1)
        for i, (action, next_action, terminal) in enumerate(updates[:-1]):
            self.assertEqual(action, chosen[i])
            self.assertEqual(next_action, chosen[i + 1])
            self.assertFalse(terminal)

    def test_run_episode_changes_q_table(self):
        env = SpaceEnvironment()
        agent = SarsaAgent(env.n_states, env.n_actions, seed=2)
        agent.run_episode(env)
        self.assertTrue(np.any(agent.q_table != 0))

    def test_separate_q_tables(self):
        a, b = make_agent(), make_agent()
        a.update(0, 0, 1, 1, 0, False)
        self.assertTrue(np.all(b.q_table == 0))

    def test_learns_to_reach_goal_on_small_map(self):
        env = SpaceEnvironment(layout=["..G", ".A.", "S.."], max_steps=30)
        agent = SarsaAgent(env.n_states, env.n_actions, alpha=0.2, gamma=0.95,
                           epsilon_decay=0.99, seed=4)
        for _ in range(800):
            agent.run_episode(env)
            agent.decay_epsilon()
        state = env.state_to_index(env.reset())
        done = False
        while not done:
            state_tuple, _, done, _ = env.step(agent.greedy_action(state))
            state = env.state_to_index(state_tuple)
        self.assertEqual(env.status, "success")


if __name__ == "__main__":
    unittest.main()
