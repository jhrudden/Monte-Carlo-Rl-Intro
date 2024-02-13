from gymnasium import Env
from enum import IntEnum
from typing import Callable, Dict, List, Tuple
import numpy as np
from collections import defaultdict

from policy import create_greedy_policy

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
            tesst = "test"
            action = policy(state)
        next_state, reward, terminated, _, __ = env.step(action)
        episodes.append((state, action, reward))
        if terminated:
            break
        state = next_state
    
    return episodes[::-1] # Reverse the episode so that the first value is the terminal state

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
        G = 0
        visited = set()

        # Update the value function for each state in the episode
        states, actions, rewards = zip(*episode)
        for i, state in enumerate(states):
            G = gamma * G + rewards[i]
            if state not in visited:
                visited.add(state)
                N[state] += 1
                V[state] += (G - V[state]) / N[state]

    return V

def monte_carlo_ex_fv(env: Env, initial_policy: Callable[[Tuple[int, int, bool]], IntEnum], gamma: float = 0.9, num_episodes: int = 10_000) -> Tuple[Dict[Tuple[int, int, bool], float], Dict[Tuple[int, int, bool], IntEnum]]:
    """
    Estimate the value function of a given policy using Monte Carlo with exploring starts. Uses first-visit policy evaluation.

    Args:
        env: The environment to evaluate the policy on.
        initial_policy: The initial policy to follow.
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
    policy = create_greedy_policy(Q) # Give the policy access to the action-value function 
    # as it will be updated during the episode

    for _ in range(num_episodes):
        # Generate an episode
        episode = generate_episode(env, policy, exploring_starts=True)
        G = 0
        visited = set() # Keep track of visited state-action pairs

        # Update the value function for each state in the episode
        states, actions, rewards = zip(*episode)
        for i, state in enumerate(states):
            action = actions[i]
            state_action = (state, action)
            G = gamma * G + rewards[i]
            if state_action not in visited:
                visited.add(state_action)
                N[state_action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state_action]
    
    # derive the value function from the action-value function
    V = {state: max(Q[state]) for state in Q}

    return V, policy


    

    