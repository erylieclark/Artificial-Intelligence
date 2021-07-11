# Name:         Erin Rylie Clark
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Tile Driver II
# Term:         Summer 2021

import random
from typing import Callable, List, Tuple

import tiledriver


def is_solvable(tiles: Tuple[int, ...]) -> bool:
    """
    Return True if the given tiles represent a solvable puzzle and False
    otherwise.

    >>> is_solvable((3, 2, 1, 0))
    True
    >>> is_solvable((0, 2, 1, 3))
    False
    """
    _, inversions = _count_inversions(list(tiles))
    width = int(len(tiles) ** 0.5)
    if width % 2 == 0:
        row = tiles.index(0) // width
        return (row % 2 == 0 and inversions % 2 == 0 or
                row % 2 == 1 and inversions % 2 == 1)
    else:
        return inversions % 2 == 0


def _count_inversions(ints: List[int]) -> Tuple[List[int], int]:
    """
    Count the number of inversions in the given sequence of integers (ignoring
    zero), and return the sorted sequence along with the inversion count.

    This function is only intended to assist |is_solvable|.

    >>> _count_inversions([3, 7, 1, 4, 0, 2, 6, 8, 5])
    ([1, 2, 3, 4, 5, 6, 7, 8], 10)
    """
    if len(ints) <= 1:
        return ([], 0) if 0 in ints else (ints, 0)
    midpoint = len(ints) // 2
    l_side, l_inv = _count_inversions(ints[:midpoint])
    r_side, r_inv = _count_inversions(ints[midpoint:])
    inversions = l_inv + r_inv
    i = j = 0
    sorted_tiles = []
    while i < len(l_side) and j < len(r_side):
        if l_side[i] <= r_side[j]:
            sorted_tiles.append(l_side[i])
            i += 1
        else:
            sorted_tiles.append(r_side[j])
            inversions += len(l_side[i:])
            j += 1
    sorted_tiles += l_side[i:] + r_side[j:]
    return (sorted_tiles, inversions)


def create_initial_state(width: int) -> Tuple[int, ...]:
    """
    Create num random initial states, check that they are solvable, then
    send the state back
    """
    orig_state: list = list(range(0,width**2))
    
    while(1):
        new_state: list = random.sample(orig_state, len(orig_state))
        if is_solvable(new_state):
            break

    return new_state


def compare_LCs(best_tiles: Tuple[int, ...], new_tiles: Tuple[int, ...], \
best_lc: int, change: int, width: int) -> Tuple[Tuple[int, ...], int, int]:
    """
    Get the set of new states based on which directions a tile can be moved.
    Return a dictionary containing each of the new states.
    """
    best_state: Tuple[int, ...] = best_tiles
    # Get the new number of linear conflicts
    lc = tiledriver.Heuristic._get_linear_conflicts(new_tiles, width)
    if (lc - best_lc) > change:
        change = lc - best_lc
    if lc > best_lc:
        best_lc = lc
        best_state = new_tiles
    return best_state, best_lc, change 

def next_states(tiles: Tuple[int, ...], width: int, best_lc: int) -> \
Tuple[Tuple[int, ...], int, int]:
    """
    Get the set of new states based on which directions a tile can be moved.
    Return a dictionary containing each of the new states.
    """
    pos: int = tiles.index(0)   # Index of the empty spot in list
    x: int = pos % width        # x position with respect to grid
    y: int = pos // width       # y position with respect to grid
    lc: int = 0                 # Number of linear conflicts for new state
    change: int = -2*((width - 1)**2 + (width - 2))
    best_state: Tuple[int,...] = [] # Set to nothing, if this is passed back,
        # we know we are either at the top or on a plateau

    # Compare x and y to the width to see if it can go up, down, left or right
    if x < width - 1:       # Possible to move left
        new = list(tiles)   # Turn Tuple into a list to modify
        new[x + y*width] = new[x + y*width + 1] # Move # to 0 spot
        new[x + y*width + 1] = 0        # Replace # with empty spot
        best_state, best_lc, change = compare_LCs(tiles, new, best_lc, change,\
            width)
    if x > 0: # Possible to move right 
        new = list(tiles)   # Turn Tuple into a list to modify
        new[x + y*width] = new[x + y*width - 1] # Move # to 0 spot
        new[x + y*width - 1] = 0        # Replace # with empty spot
        best_state, best_lc, change = compare_LCs(tiles, new, best_lc, change,\
            width)
    if y < width - 1: # Possible to move up 
        new = list(tiles)   # Turn Tuple into a list to modify
        new[x + y*width] = new[x + y*width + width] # Move # to 0 spot
        new[x + y*width + width] = 0    # Replace # with empty spot
        best_state, best_lc, change = compare_LCs(tiles, new, best_lc, change,\
            width)
    if y > 0: # Possible to move down 
        new = list(tiles)   # Turn Tuple into a list to modify
        new[x + y*width] = new[x + y*width - width] # Move # to 0 spot
        new[x + y*width - width] = 0    # Replace # with empty spot
        best_state, best_lc, change = compare_LCs(tiles, new, best_lc, change,\
            width)
    return best_state, best_lc, change


def conflict_tiles(width: int, min_lc: int) -> Tuple[int, ...]:
    """
    Create a solvable shuffled puzzle of the given width with a minimum number
    of linear conflicts (ignoring Manhattan distance).
    Using hill climbing.
    >>> tiles = conflict_tiles(3, 5)
    >>> tiledriver.Heuristic._get_linear_conflicts(tiles, 3)
    5
    """
    best_state: list = []   # Current best state
    best_lc: int = 0       # Number of linear conflicts of best state
    max_lc: int = 2*((width - 1)**2 + (width - 2))
    change: int = -1 
    max_plateau: int = 20
    k: int = 0
    print("Max Conflicts Possible: ", max_lc)
    while(1):
        # Get a new random state
        if change < 0:
            new_state = tuple(create_initial_state(width))
            k = 0
            #print("Restarting...")
        elif change == 0:
            k = k + 1
            #print("Plateau...")
            if k > max_plateau:
                new_state = tuple(create_initial_state(width))
                k = 0
        else:
            k = 0
        # Only use this state if it has more than third req linear conflicts
        lc = tiledriver.Heuristic._get_linear_conflicts(new_state, width)
        if lc < max_lc/3:   # Find a better place to start
            continue
        elif lc >= min_lc:  # In case we get lucky with min linear conflicts
            print("Got Lucky")
            print("Final State: ", new_state)
            print("Linear Conflicts: ", lc)
            return new_state
        # Otherwise, get the next possible states
        new_state, best_lc, change = next_states(new_state, width, best_lc)
        if best_lc >= min_lc:
            break
    
    print("Final State: ", new_state)
    print("Linear Conflicts: ", best_lc)
    return new_state


def shuffle_tiles(width: int, min_len: int,
                  solve_puzzle: Callable[[Tuple[int, ...]], str]
) -> Tuple[int, ...]:
    """
    Create a solvable shuffled puzzle of the given width with an optimal
    solution length equal to or greater than the given minimum length.

    >>> tiles = shuffle_tiles(3, 6, tiledriver.solve_puzzle)
    >>> len(tiledriver.solve_puzzle(tiles))
    6
    """


def main() -> None:
    conflict_tiles(5,18)
    pass  # optional program test driver


if __name__ == "__main__":
    main()
