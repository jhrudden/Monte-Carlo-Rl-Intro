from gymnasium import Env
from enum import IntEnum
from typing import Callable, Dict, List, Tuple
import numpy as np
from collections import defaultdict

from policy import BlackJackPolicy

def generate_episode(env: Env, policy: Callable[[Tuple[int, int, bool]], IntEnum], exploring_starts: bool = False) -> List[Tuple[Tuple[int, int, bool], IntEnum, float]]:
    """
    Generate an episode using the given policy.

    Args:
        env: The environment to sample episodes from.
        policy: The policy to follow.
        exploring_starts: Whether to use exploring starts to sample the initial state and action.

    Returns:
        A list of (state, action, reward) tuples.
    """
    state, _ = env.reset()
    episodes = []
    while True:
        if len(episodes) == 0 and exploring_starts:
            action = env.action_space.sample()
        else:
            action = policy(state)
        next_state, reward, terminated, _, __ = env.step(action)
        episodes.append((state, action, reward))
        if terminated:
            break
        state = next_state
    
    return episodes

def get_first_visit_indices(episode: List[Tuple[Tuple[int, int, bool], IntEnum, float]]) -> Dict[Tuple[int, int, bool], int]:
    """
    Get the time step of the first visit to each state in the episode.

    Args:
        episode: The episode to get the first visit indices for.

    Returns:
        A dictionary mapping state to the index of the first visit to that state.
    """
    first_visit_step = {}

    for i, (state, _, _) in enumerate(episode):
        if state not in first_visit_step:
            first_visit_step[state] = i

    return first_visit_step

def monte_carlo_prediction_fv(env: Env, policy: Callable[[Tuple[int, int, bool]], IntEnum], gamma: float = 0.9, num_episodes: int =10_000) -> Dict[Tuple[int, int, bool], float]:
    """
    Estimate the value function of a given policy using first-visit Monte Carlo policy evaluation.

    Args:
        env: The environment to evaluate the policy on.
        policy: The policy to evaluate.
        num_episodes: The number of episodes to sample.

    Returns:
        A dictionary mapping state to their estimated value.
    """
    # Initialize the value function and the count of visits to each state
    V = defaultdict(float)
    N = defaultdict(int)

    for _ in range(num_episodes):
        # Generate an episode
        episode = generate_episode(env, policy)
        first_visit_step = get_first_visit_indices(episode)
        
        G = 0
        T = len(episode)
        for t in range(T-1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if first_visit_step[state] == t:
                N[state] += 1
                V[state] += (G - V[state]) / N[state]

    return V

def monte_carlo_ex_fv(env: Env, initial_policy_builder: Callable, gamma: float = 0.9, num_episodes: int = 10_000) -> Tuple[Dict[Tuple[int, int, bool], float], Dict[Tuple[int, int, bool], IntEnum]]:
    """
    Estimate the value function of a given policy using Monte Carlo with exploring starts. Uses first-visit policy evaluation.

    Args:
        env: The environment to evaluate the policy on.
        initial_policy_builder: The initial policy to follow. Should be a function that takes Q-values and returns a policy.
        gamma: The discount factor.
        num_episodes: The number of episodes to sample.

    Returns:
        A tuple containing:
            - A dictionary mapping state to their estimated value.
            - A dictionary mapping state to optimal action.
    """
    # Initialize the value function and the count of visits to each state
    N = defaultdict(int)
    Q = defaultdict(lambda: list(np.zeros(env.action_space.n)))
    policy = initial_policy_builder(Q) # Give the policy access to the action-value function 
    # as it will be updated during the episode

    for _ in range(num_episodes):
        # Generate an episode
        episode = generate_episode(env, policy, exploring_starts=True)
        first_visit_step = get_first_visit_indices(episode)

        G = 0
        T = len(episode)
        for t in range(T - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if first_visit_step[state] == t:
                state_action = (state, action)
                N[state_action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state_action]
    
    # derive the value function from the action-value function
    V = {state: max(Q[state]) for state in Q}

    return V, policy


    

    