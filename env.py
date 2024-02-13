import gymnasium as gym
from enum import IntEnum
from gymnasium import Env

class BlackjackAction(IntEnum):
    STICK = 0
    HIT = 1

def create_blackjack_env():
    return gym.make('Blackjack-v1', sab=True)



