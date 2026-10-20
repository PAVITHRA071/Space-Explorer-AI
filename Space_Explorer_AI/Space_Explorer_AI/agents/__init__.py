"""Reinforcement-learning agents."""

from config import Q_LEARNING, SARSA

from .base_agent import TabularAgent, episode_result  # noqa: F401
from .q_learning import QLearningAgent
from .sarsa import SarsaAgent

AGENT_CLASSES = {Q_LEARNING: QLearningAgent, SARSA: SarsaAgent}


def create_agent(algorithm, n_states, n_actions, hyperparameters, seed=None):
    """Create a fresh agent for ``algorithm`` ("q_learning" or "sarsa")."""
    try:
        agent_class = AGENT_CLASSES[algorithm]
    except KeyError:
        raise ValueError(f"Unknown algorithm {algorithm!r}.") from None
    return agent_class(n_states, n_actions, seed=seed, **hyperparameters)
