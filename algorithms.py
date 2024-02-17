from gymnasium import Env
from enum import IntEnum
from typing import Callable, Dict, List, Tuple
import numpy as np
from collections import defaultdict
from tqdm import trange

from policy import BlackJackPolicy, EpsilonPolicy

def generate_episode(env: Env, policy: Callable[[Tuple[int, int, bool]], Tuple[IntEnum, float]], exploring_starts: bool = False) -> List[Tuple[Tuple[int, int, bool], float, IntEnum, float]]:
    """
    Generate an episode using the given policy.

    Args:
        env: The environment to sample episodes from.
        policy: The policy to follow.
        exploring_starts: Whether to use exploring starts to sample the initial state and action.

    Returns:
        A list of (state, action, action_prob, reward) tuples.
    """
    state, _ = env.reset()
    episodes = []
    while True:
        if len(episodes) == 0 and exploring_starts:
            action = env.action_space.sample()
            action_prob = 1 / env.action_space.n
        else:
            action, action_prob = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        episodes.append((state, action, action_prob, reward))
        if terminated or truncated:
            break
        state = next_state
    
    return episodes

# TODO: make get_first_visit format episode step based on some callable
def get_first_visit_indices(visits: List) -> Dict[Tuple[int, int, bool], int]:
    """
    Get the time step of the first visit to each state in the episode.

    Args:
        episode: The episode to get the first visit indices for.
        with_action: Whether to include the action in the state.

    Returns:
        A dictionary mapping state to the index of the first visit to that state.
    """
    first_visit_step = {}

    for i, v in enumerate(visits):
        if v not in first_visit_step:
            first_visit_step[v] = i

    return first_visit_step

def on_policy_monte_carlo_prediction_fv(env: Env, policy: Callable[[Tuple[int, int, bool]], IntEnum], gamma: float = 0.9, num_episodes: int =10_000) -> Dict[Tuple[int, int, bool], float]:
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

    for _ in trange(num_episodes):
        # Generate an episode
        episode = generate_episode(env, policy)
        visits = [v[0] for v in episode]
        first_visit_step = get_first_visit_indices(visits)
        
        G = 0
        T = len(episode)
        for t in range(T-1, -1, -1):
            state, action, _, reward = episode[t]
            G = gamma * G + reward
            if first_visit_step[state] == t:
                N[state] += 1
                V[state] += (G - V[state]) / N[state]

    return V

def on_policy_monte_carlo_fv(env: Env, initial_policy_builder: Callable, gamma: float = 0.9, num_episodes: int = 10_000, es: bool = False) -> Tuple[Dict[Tuple[int, int, bool], float], Dict[Tuple[int, int, bool], IntEnum]]:
    """
    Estimate the value function of a given policy using on policy Monte Carlo control with exploring starts. Uses first-visit policy evaluation.

    Args:
        env: The environment to evaluate the policy on.
        initial_policy_builder: The initial policy to follow. Should be a function that takes Q-values and returns a policy.
        gamma: The discount factor.
        num_episodes: The number of episodes to sample.
        es: Whether to use exploring starts to sample the initial state and action.

    Returns:
        A tuple containing:
            - A dictionary mapping state to their estimated value.
            - A dictionary mapping state to optimal action.
    """
    # Initialize the value function and the count of visits to each state
    N = defaultdict(lambda: list(np.zeros(env.action_space.n)))
    Q = defaultdict(lambda: list(np.zeros(env.action_space.n)))
    total_returns = np.zeros(num_episodes)
    policy = initial_policy_builder(Q) # Give the policy access to the action-value function 
    # as it will be updated during the episode

    for i in trange(num_episodes):
        # Generate an episode
        episode = generate_episode(env, policy, exploring_starts=es)
        visits = [(v[0], v[1]) for v in episode]
        first_visit_tstep = get_first_visit_indices(visits)

        G = 0
        T = len(episode)
        for t in range(T - 1, -1, -1):
            state, action, _, reward = episode[t]
            G = gamma * G + reward
            if first_visit_tstep[(state, action)] == t:
                N[state][action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state][action]
        
        total_returns[i] = G
    
    # derive the value function from the action-value function
    V = {state: max(Q[state]) for state in Q}

    return V, policy, total_returns

def off_policy_monte_carlo(env: Env, behavior_policy_builder: Callable[[Tuple[int, int, bool]], IntEnum], gamma: float = 0.9, num_episodes: int = 10_000) -> Dict[Tuple[int, int, bool], float]: 
    """
    Estimate the value function of a target policy using off-policy Monte Carlo control.

    Args:
        env: The environment to evaluate the policy on.
        behavior_policy_builder: The behavior policy to follow. Should be a function that takes Q-values and returns a policy.
        gamma: The discount factor.
        num_episodes: The number of episodes to sample.
    """
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.random.rand(env.action_space.n))

    target_policy = EpsilonPolicy.create_greedy_policy(Q)
    behavior_policy = behavior_policy_builder(Q)

    total_returns = np.zeros((2, num_episodes))

    for i in trange(num_episodes):
        episode = generate_episode(env, behavior_policy, exploring_starts=False)
        G = 0
        W = 1
        update = True
        for t in range(len(episode) - 1, -1, -1):
            state, action, action_prob_b, reward = episode[t]
            G  = gamma * G + reward
            if not update:
                pass
            else:
                C[state][action] += W
                Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
                t_action, action_prob_t = target_policy(state)
                if action != t_action:
                    update = False
                    # Since target policy is deterministic, we can break if the action behavior policy took is not the same as the target policy
                    # We can't do this if the target policy is stochastic
                else:
                    W *= 1 / action_prob_b  # W = W * (pi(A|S) / b(A|S)) = W * (1 / b(A|S)) since pi(A|S) = 1
        total_returns[0][i] = G

        # get return for target policy
        episode = generate_episode(env, target_policy, exploring_starts=False)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            _, _, _, reward = episode[t]
            G = gamma * G + reward
        total_returns[1][i] = G
    
    V = {state: max(Q[state]) for state in Q}
    return V, target_policy, total_returns


    