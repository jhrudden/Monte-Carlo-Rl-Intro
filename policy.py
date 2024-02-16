from typing import List, Tuple, Dict, Sequence
from env import BlackjackAction, FourRoomAction
import numpy as np

class BlackJackPolicy:
    @staticmethod
    def default_blackjack_policy(observation: Tuple[int, int, bool]) -> BlackjackAction:
        """
        A simple policy for the Blackjack environment. Stick if the score is 20 or higher, hit otherwise.

        Args:
            observation: A tuple containing the player's score, the dealer's score, and whether the player has a usable ace.
        """
        score, _, _ = observation
        if score == 20 or score == 21:
            return BlackjackAction.STICK
        else:
            return BlackjackAction.HIT
    
    @staticmethod
    def create_greedy_policy(Q: Dict[Tuple, List[float]]) -> BlackjackAction:
        """
        A policy that selects the action that maximizes the Q-value for the given observation.
        If the Q-values are equal, the policy will choose based on default_blackjack_policy.

        Args:
            Q: dictionary mapping state to their Q-values.
        """

        def get_action(state : Tuple[int, int, bool]) -> BlackjackAction:
            nonlocal Q
            if state in Q:
                optimal_action = np.argmax(Q[state])
                return BlackjackAction(optimal_action)
            else:
                return BlackJackPolicy.default_blackjack_policy(state)

        return get_action


def argmax(arr: Sequence[float]) -> int:
    """
    Argmax that breaks ties randomly (np.argmax only returns first index in case of ties, we don't like this)

    Args:
        arr: sequence of values
    
    Returns:
        index of maximum value
    """
    max_val = np.max(arr)
    max_indices = np.where(arr == max_val)[0]
    return np.random.choice(max_indices)

class EpsilonPolicy:
    # 4 room environment
    @staticmethod
    def create_epsilon_soft_policy(Q: Dict[Tuple, List[float]], epsilon: float, na: int) -> int:
        """
        A policy that selects the action that maximizes the Q-value for the given observation with probability 1 - epsilon,

        Args:
            Q: dictionary mapping state to their Q-values.
            epsilon: the probability to select a random action
            na: number of actions
        """

        def get_action(state: Tuple) -> int:
            random_probs = np.zeros(na) + epsilon / na
            optimal_action = argmax(Q[state])
            random_probs[optimal_action] += 1 - epsilon
            return np.random.choice(na, p=random_probs)
        return get_action
    
    @staticmethod
    def create_greedy_policy(Q: Dict[Tuple, List[float]]) -> int:
        """
        A policy that selects the action that maximizes the Q-value for the given observation.

        Args:
            Q: dictionary mapping state to their Q-values.
        """

        def get_action(state: Tuple) -> int:
            return argmax(Q[state]) # ties are broken consistently instead of randomly as in create_epsilon_soft_policy
        
        raise NotImplementedError("Policies should return action and the probability of taking that action. For our current policie")
        return get_action