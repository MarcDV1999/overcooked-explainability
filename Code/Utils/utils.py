"""
Useful functions.
"""
from enum import Enum

import numpy as np


class Actions(Enum):
    Top = 0
    Bottom = 1
    Right = 2
    Left = 3
    Stay = 4
    Interact = 5


class Orientations(Enum):
    Top = (0, -1)
    Bottom = (0, 1)
    Right = (1, 0)
    Left = (-1, 0)


class CardinalDirection(Enum):
    North = 'N'
    NorthEast = 'NE'
    East = 'E'
    SouthEast = 'SE'
    South = 'S'
    SouthWest = 'SW'
    West = 'W'
    NorthWest = 'NW'


def get_move(start_pos, end_pos):
    if start_pos[0] < end_pos[0]:
        return Actions.Right
    if start_pos[0] > end_pos[0]:
        return Actions.Left
    if start_pos[1] < end_pos[1]:
        return Actions.Bottom
    if start_pos[1] > end_pos[1]:
        return Actions.Top
    else:
        return Actions.Stay


def action_num_to_char(action_num):
    """
    Converts an action (int) to a char
    """

    if action_num == 0:
        return "↑"
    elif action_num == 1:
        return '↓'
    elif action_num == 2:
        return '→'
    elif action_num == 3:
        return '←'
    # Stay
    elif action_num == 4:
        return 's'
    # Interact
    elif action_num == 5:
        return 'x'


def action_num_to_str(action_num):
    """
    Converts the action (int) in a str
    """
    if action_num == 0:
        return "top"
    elif action_num == 1:
        return 'bottom'
    elif action_num == 2:
        return 'right'
    elif action_num == 3:
        return 'left'
    elif action_num == 4:
        return 'stay'
    elif action_num == 5:
        return 'interact'


def get_assigned_color(action_num):
    """
    Returns the color assigned to the action (int).
    """
    if action_num == 0:
        return "FF6B6B"
    elif action_num == 1:
        return '4D96FF'
    elif action_num == 2:
        return '6BCB77'
    elif action_num == 3:
        return 'FFD93D'
    # Stay
    elif action_num == 4:
        return '525E75'
    # Interact
    elif action_num == 5:
        return 'FF7BA9'


def get_weight_assigned_color(weight):
    """
    Returns the color that represents the given weight
    """
    if weight >= 0.75:
        return '#332FD0'
    if weight >= 0.5:
        return '#9254C8'
    if weight >= 0.25:
        return '#E15FED'
    else:
        return '#6EDCD9'


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        return v
    return v / norm
