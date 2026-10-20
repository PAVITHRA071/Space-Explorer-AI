"""SARSA (on-policy temporal-difference control).

Update rule
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]

The name comes from the quintuple used in every update:
State, Action, Reward, next State, next Action.

The target uses a' - the action that the epsilon-greedy policy has ACTUALLY
chosen for the next step (and that the agent then really takes).  Because the
update includes the agent's own exploratory moves, SARSA learns the value of
the policy it is following (*on-policy*), which usually makes it more careful
near dangers such as asteroids.
"""

from .base_agent import TabularAgent, episode_result


class SarsaAgent(TabularAgent):
    name = "SARSA"
    key = "sarsa"

    def update(self, state, action, reward, next_state, next_action, terminal):
        """Apply one SARSA update and return the TD error.

        ``next_action`` is the action already selected for ``next_state``.
        For a terminal state there is no next action and no bootstrapping.
        """
        current = self.q_table[state, action]
        if terminal:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state, next_action]
        td_error = target - current
        self.q_table[state, action] = current + self.alpha * td_error
        return td_error

    def run_episode(self, env):
        """Play (and learn from) one full episode; return its statistics."""
        state = env.state_to_index(env.reset())
        action = self.choose_action(state)          # first action, epsilon-greedy
        done = False
        while not done:
            next_state_tuple, reward, done, info = env.step(action)
            next_state = env.state_to_index(next_state_tuple)
            if info["terminated"]:
                # Reached the planet or crashed: no next action exists.
                self.update(state, action, reward, next_state, None, True)
            else:
                # Choose a' with the current policy FIRST, use it in the
                # update, and then really perform it on the next loop pass.
                next_action = self.choose_action(next_state)
                self.update(state, action, reward, next_state, next_action, False)
                action = next_action
            state = next_state
        return episode_result(env)
