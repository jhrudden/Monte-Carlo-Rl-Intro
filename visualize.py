import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple
from matplotlib.patches import Patch

from env import BlackjackAction

def plot_blackjack_value_function(ax: Axes3D, V: Dict[Tuple[int, int, bool], float], usable_ace: bool, title: str):
    """
    Plots the value function for Blackjack as a surface plot on the given axes.

    This function is designed to reproduce a part of Figure 5.1 from the book
    "Introduction to Reinforcement Learning". It plots the value function of a Blackjack policy
    for either situations with a usable ace or without a usable ace.
    
    Parameters:
    ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D subplot axes to plot on.
    V (dict): A dictionary where the keys are tuples of the form (player_sum, dealer_card, usable_ace)
              and the values are the estimated values.
    usable_ace (bool): True if the plot is for states with a usable ace, False otherwise.
    title (str): The title of the subplot.
    """
    elev_angle = 30  # Adjust this angle to change the elevation
    azim_angle = -60  # Adjust this angle to change the azimuth

    X, Y = np.meshgrid(range(1, 11), range(12, 22))  # Dealer showing, Player's sum
    Z = np.apply_along_axis(lambda idx: V[(idx[1], idx[0], usable_ace)], 2, np.dstack([X, Y]))

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolors='k', alpha=0.3)
    ax.set_title(title)
    ax.set_ylabel('Player Score')
    ax.set_xlabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_zticks(range(-1, 2))
    ax.set_zticklabels(['-1', '', '1'])
    # dealer label A to 10
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(['A'] + [str(n) for n in range(2, 10)] + ['10'])
    # player label 12 to 21
    ax.set_yticks(range(12, 22))
    ax.set_yticklabels(range(12, 22))
    ax.view_init(elev=elev_angle, azim=azim_angle)

def plot_blackjack_policy(ax: plt.Axes, policy: Dict[Tuple[int, int, bool], int], usable_ace: bool, title: str):
    """
    Plots the policy for Blackjack, showing when to hit or stick.

    Parameters:
    ax (matplotlib.axes._subplots.AxesSubplot): The 2D subplot axes to plot on.
    policy (dict): A dictionary mapping states (player_sum, dealer_card, usable_ace) to actions (IntEnum).
    usable_ace (bool): True if the policy is for situations with a usable ace, False otherwise.
    title (str): The title of the subplot.
    """
    # Create a 10x10 grid to represent the policy for player sum 12-21 and dealer card A-10
    policy_grid = np.zeros((10, 10))
    
    # Fill the policy grid with the action values from the policy
    for player_sum in range(12, 22):
        for dealer_card in range(1, 11):
            action = policy((player_sum, dealer_card, usable_ace))
            policy_grid[player_sum - 12, dealer_card - 1] = action 
    
    policy_grid = np.vstack([np.ones(10), policy_grid])
    # We need to flip the policy array vertically because the plot shows the y-axis in ascending order
    policy_grid_flipped = np.flipud(policy_grid)
    
    # Generate a meshgrid for the policy matrix dimensions
    X, Y = np.meshgrid(np.arange(0.5, 10.5), np.arange(10.5, 21.5))    

    cmap = plt.get_cmap('gray', 2)
    
    # Plot the policy using imshow with the correct extent
    im = ax.imshow(policy_grid_flipped, cmap=cmap, extent=[0, 10, 10, 21], aspect='equal', alpha=0.2, origin='upper')

    # Draw the contour line where the policy changes from hit (0) to stick (1)
    # cs = ax.contour(X, Y, policy_grid, levels=[0.5], colors='black', linewidths=1)
    # ax.clabel(cs, fmt=lambda x: 'STICK' if x > 0.5 else 'HIT', fontsize=9, inline=True)
    ax.set_aspect('equal')
    
    # Formatting the plot
    ax.set_title(title)
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_xticks(np.arange(1, 11))
    ax.set_yticks(np.arange(10, 22)) # Ensure that ticks are set from 11 to 21
    ax.set_xticklabels(['A'] + [str(i) for i in range(2, 11)])
    ax.set_yticklabels([str(i) for i in range(10, 22)]) 
    # put y axis on right side
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    
    # remove top and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # add legend for imshow colors 
    legend_elements = [Patch(facecolor='gray', edgecolor='gray', label='HIT'),
                   Patch(facecolor='white', edgecolor='gray', label='STICK')]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), frameon=False, fontsize=20)


