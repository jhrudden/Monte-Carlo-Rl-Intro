import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
from enum import IntEnum
from typing import Tuple, Optional, List
import numpy as np
from numpy import ndarray
from utils import line_cross

def register_env(id: str, entry_point: str, max_episode_steps: Optional[int] = None):
    """Register custom gym environment so that we can use `gym.make()`
    Note: the max_episode_steps option controls the time limit of the environment.
    You can remove the argument to make FourRooms run without a timeout.
    """
    register(id=id, entry_point=entry_point, max_episode_steps=max_episode_steps)

class BlackjackAction(IntEnum):
    STICK = 0
    HIT = 1

def create_blackjack_env():
    return gym.make('Blackjack-v1', sab=True)


def get_four_rooms_env(goal_pos=(10, 10)):
    """
    Get the FourRooms environment
    Args:
        goal_pos (Tuple[int, int]): goal position
    Returns:
        env (FourRoomsEnv): FourRooms environment
    """
    try:
        spec = gym.spec('FourRooms-v0')
    except:
        register_env("FourRooms-v0", "env:FourRoomsEnv", max_episode_steps=459)
    finally:
        return gym.make('FourRooms-v0', goal_pos=goal_pos)


class FourRoomAction(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

def actions_to_dxdy(action: FourRoomAction) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        FourRoomAction.LEFT: (-1, 0),
        FourRoomAction.DOWN: (0, -1),
        FourRoomAction.RIGHT: (1, 0),
        FourRoomAction.UP: (0, 1),
    }
    return mapping[action]

def perpendicular_actions(action: FourRoomAction) -> List[FourRoomAction]:
    """
    Helper function to get the perpendicular actions to the given action
    Args:
        action (Action): taken action
    Returns:
        perpendicular_actions (List[Action]): Perpendicular actions to the given action
    """
    mapping = {
        FourRoomAction.LEFT: [FourRoomAction.DOWN, FourRoomAction.UP],
        FourRoomAction.DOWN: [FourRoomAction.LEFT, FourRoomAction.RIGHT],
        FourRoomAction.RIGHT: [FourRoomAction.DOWN, FourRoomAction.UP],
        FourRoomAction.UP: [FourRoomAction.LEFT, FourRoomAction.RIGHT],
    }
    return mapping[action]


class FourRoomsEnv(Env):
    """Four Rooms gym environment.

    This is a minimal example of how to create a custom gym environment. By conforming to the Gym API, you can use the same `generate_episode()` function for both Blackjack and Four Rooms envs.
    """

    def __init__(self, goal_pos=(10, 10)) -> None:
        super().__init__()
        self.rows = 11
        self.cols = 11

        # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
        self.walls = [
            (0, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 0),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 9),
            (5, 10),
            (6, 4),
            (7, 4),
            (9, 4),
            (10, 4),
        ]

        self.start_pos = (0, 0)
        self.goal_pos = goal_pos
        self.agent_pos = None

        self.action_space = spaces.Discrete(len(FourRoomAction))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

    def reset(self) -> Tuple[int, int]:
        """Reset agent to the starting position.

        Returns:
            observation (Tuple[int,int]): returns the initial observation
        """
        self.agent_pos = self.start_pos

        return (self.agent_pos, {})

    def step(self, action: FourRoomAction) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 for more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """
        EPSILON = 0.1
        
        random_val = np.random.random()
        if random_val < EPSILON:
            action_taken = np.random.choice(perpendicular_actions(action))
        else:
            action_taken = action

        dx, dy = actions_to_dxdy(action_taken)
        next_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        if self._valid_position(next_pos):
            self.agent_pos = next_pos

        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 1.0
        else:
            done = False
            reward = 0.0

        return self.agent_pos, reward, done, False, {}
    
    def _valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Helper function to check if a position is valid
        Args:
            pos (Tuple[int, int]): position to check
        Returns:
            valid (bool): True if position is valid, False otherwise
        """
        return pos not in self.walls and 0 <= pos[0] < self.cols and 0 <= pos[1] < self.rows


class TrackType(IntEnum):
    Track0 = 0
    Track1 = 1

def get_race_track_env(track_type: TrackType = TrackType.Track0):
    """
    Get the FourRooms environment
    Args:
        goal_pos (Tuple[int, int]): goal position
    Returns:
        env (FourRoomsEnv): FourRooms environment
    """
    try:
        spec = gym.spec('RaceTrack-v0')
    except:
        register_env("RaceTrack-v0", "env:RaceTrackEnv", max_episode_steps=459)
    finally:
        return gym.make('RaceTrack-v0', track_type)

def get_track(track_type: TrackType) -> ndarray:
    if track_type == TrackType.Track0:
        return track0()
    elif track_type == TrackType.Track1:
        return track1()
    else:
        raise ValueError("Invalid track type")

class RaceTrackEnv(Env):
    def __init__(self, track_type: TrackType = TrackType.Track0, max_speed: int = 4) -> None:
        super().__init__()
        self.track = get_track(track_type)
        self.rows = len(self.track)
        self.cols = len(self.track[0])

        self.max_speed = max_speed
        self.action_space = spaces.Tuple(
            (spaces.Discrete(3, start=-1), spaces.Discrete(3, start=-1))
        )
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(len(self.track)), spaces.Discrete(len(self.track[0])), spaces.Discrete(self.max_speed + 1), spaces.Discrete(self.max_speed + 1))
        )

        # finding the starting positions
        start_rows, start_cols = np.where(self.track == 2)
        self.start_positions = list(zip(start_rows, start_cols))

        finish_rows, finish_cols = np.where(self.track == 3)
        self.finish_positions = list(zip(finish_rows, finish_cols))

        self.agent_pos = None
        self.agent_velocity = (0, 0)


    def reset(self) -> Tuple[int, int]:
        """Reset agent to the starting position.

        Returns:
            observation (Tuple[int,int]): returns the initial observation
        """
        # choose a random starting position
        self.agent_pos = self._choose_starting_position()
        self.agent_velocity = (0, 0)

        return ((*self.agent_pos, *self.agent_velocity), {})

    def step(self, action: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        """
        dx, dy = action

        # check if the action is valid
        if not (-1 <= dx < 1 and 1 <= dy < -1):
            raise ValueError("Invalid action")
        
        reward = -1
        done = False

        # stochasticity
        if np.random.random() <= 0.1:
            dx, dy = 0, 0
        
        # update the velocity
        self.agent_velocity[0] = max(0, min(self.agent_velocity[0] + dx, self.max_speed))
        self.agent_velocity[1] = max(0, min(self.agent_velocity[1] + dy, self.max_speed))

        new_pos = (self.agent_pos[0] + self.agent_velocity[0], self.agent_pos[1] + self.agent_velocity[1])
        if self._finish_line(new_pos):
            done = True
            reward = 0

        elif not self._valid_position(new_pos):
            self.agent_velocity = (0, 0)
            new_pos = self._choose_starting_position()
        
        else:
            self.agent_pos = new_pos
        
        return (*self.agent_pos, *self.agent_velocity), reward, done, False, {}

    def _finish_line(self, new_pos: Tuple[int, int]) -> bool:
        """
        Check if path from current position to new position crosses the finish line
        Args:
            new_pos (Tuple[int, int]): new position
        Returns:
            crossed (bool): True if finish line is crossed, False otherwise
        """
        # check if any of any of the finish positions cross vector from current position to new position
        # get goal state with min and max y
        goal_1 = self.finish_positions[0]
        goal_last = self.finish_positions[-1]

        # TODO: rewrite if goal lines are note rightmost elements of the track
        # check agent is either on or to the right of the finish line
        if new_pos[1] <= goal_1[1]:
            return False
        
        return line_cross(self.agent_pos, new_pos, goal_1, goal_last)

    
    def _valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Helper function to check if a position is valid
        Args:
            pos (Tuple[int, int]): position to check
        Returns:
            valid (bool): True if position is valid, False otherwise
        """
        return 0 <= pos[0] < len(self.cols) and 0 <= pos[1] < len(self.rows) and self.track[pos] != 1
    
    def _choose_starting_position(self) -> Tuple[int, int]:
        """
        Helper function to choose a random starting position
        Returns:
            starting_position (Tuple[int, int]): random starting position
        """
        return self.start_positions[np.random.choice(len(self.start_positions))]


        
        




def track0():
    return np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
      ], dtype=np.int32)

def track1():
    return np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      ], dtype=np.int32)
