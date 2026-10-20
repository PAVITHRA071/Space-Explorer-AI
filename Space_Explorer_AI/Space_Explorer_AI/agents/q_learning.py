"""Q-Learning (off-policy temporal-difference control).

Update rule
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

The target uses the BEST action in the next state, whatever the agent will
really do next.  That is why Q-Learning is called *off-policy*: it learns the
value of the greedy policy while still exploring with epsilon-greedy.
"""

import numpy as np

from .base_agent import TabularAgent, episode_result


class QLearningAgent(TabularAgent):
    name = "Q-Learning"
    key = "q_learning"

    def update(self, state, action, reward, next_state, terminal):
        """Apply one Q-Learning update and return the TD error.

        ``terminal`` is True when the episode ended because the ship reached
        the planet or crashed.  A terminal state has no future, so we do NOT
        bootstrap from it: the target is just the reward.
        """
        current = self.q_table[state, action]
        if terminal:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        td_error = target - current
        self.q_table[state, action] = current + self.alpha * td_error
        return td_error

    def run_episode(self, env):
        """Play (and learn from) one full episode; return its statistics."""
        state = env.state_to_index(env.reset())
        done = False
        while not done:
            action = self.choose_action(state)                 # epsilon-greedy
            next_state_tuple, reward, done, info = env.step(action)
            next_state = env.state_to_index(next_state_tuple)
            # info["terminated"] is False for a step-limit timeout, because
            # the state does not contain the clock and the future still exists.
            self.update(state, action, reward, next_state, info["terminated"])
            state = next_state
        return episode_result(env)
