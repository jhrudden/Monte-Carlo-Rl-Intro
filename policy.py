from typing import List, Tuple, Dict
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

# TODO: cleanup to work with ArbitraryEnv + Action space
class FourRoomPolicy:
    # 4 room environment
    @staticmethod
    def create_epsilon_soft_policy(Q: Dict[Tuple, List[float]], epsilon: float) -> FourRoomAction:
        """
        A policy that selects the action that maximizes the Q-value for the given observation with probability 1 - epsilon,

        Args:
            Q: dictionary mapping state to their Q-values.
            epsilon: the probability to select a random action
        """

        def get_action(state: Tuple[int, int, bool]) -> FourRoomAction:
            nonlocal Q
            if np.random.random() > epsilon and state in Q:
                optimal_action = np.argmax(Q[state])
                return FourRoomAction(optimal_action)
            else:
                return np.random.choice(list(FourRoomAction))

        return get_action