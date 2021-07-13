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
    orig_state: list = list(range(0, width**2))
    length = len(orig_state)
    
    while True:
        new_state: list = random.sample(orig_state, length)
        if is_solvable(tuple(new_state)):
            break

    return tuple(new_state)


def compare_LCs(best_tiles: Tuple[int, ...], new_tiles: Tuple[int, ...], \
best_lc: int, change: int, width: int) -> Tuple[Tuple[int, ...], int, int]:
    """
    Determine if the given new state has a better linear conflict count than
    the current best state/linear conflict count. Calculate the change in
    linear conflicts to indicate if this state was better or worse than the
    previous one.
    """
    best_state: Tuple[int, ...] = best_tiles
    lc: int = 0                 # Number of linear conflicts for new state
    # Get the new number of linear conflicts
    lc = tiledriver.Heuristic._get_linear_conflicts(new_tiles, width)
    if (lc - best_lc) > change: 
        change = lc - best_lc
    if lc >= best_lc:
        best_lc = lc
        best_state = new_tiles
    return best_state, best_lc, change 


def switch_tiles(tiles: Tuple[int, ...], width: int, x: int, y: int, d: int) \
-> Tuple[int, ...]:
    """
    Use the direction to move a tile and the empty tile location to switch the
    number and the 0, then return the result.
    """
    new = list(tiles)
    if d == 0:
        new = list(tiles)   # Turn Tuple into a list to modify
        new[x + y*width] = new[x + y*width + 1] # Move # to 0 spot
        new[x + y*width + 1] = 0        # Replace # with empty spot
    elif d == 1:
        new = list(tiles)   # Turn Tuple into a list to modify
        new[x + y*width] = new[x + y*width - 1] # Move # to 0 spot
        new[x + y*width - 1] = 0        # Replace # with empty spot
    elif d == 2:
        new = list(tiles)   # Turn Tuple into a list to modify
        new[x + y*width] = new[x + y*width + width] # Move # to 0 spot
        new[x + y*width + width] = 0    # Replace # with empty spot
    elif d == 3:
        new = list(tiles)   # Turn Tuple into a list to modify
        new[x + y*width] = new[x + y*width - width] # Move # to 0 spot
        new[x + y*width - width] = 0    # Replace # with empty spot
    return tuple(new)


def next_states(tiles: Tuple[int, ...], width: int, best_lc: int, \
last_move: str) -> Tuple[Tuple[int, ...], int, int, str]:
    """
    Get the set of new states based on which directions a tile can be moved.
    Determine if the new states are better than the previously determined
    best state. Keeps track of the last move made to that we are not back-
    tracking if on a plateau.
    """
    pos: int = tiles.index(0)   # Index of the empty spot in list
    x: int = pos % width        # x position with respect to grid
    y: int = pos // width       # y position with respect to grid
    change: int = -2*((width - 1)**2 + (width - 2))
    best_state: Tuple[int, ...] = tiles # Set to current state

    # Compare x and y to the width to see if it can go up, down, left or right
    if x < width - 1:       # Possible to move left
        if last_move != "L":    # Prevent reversing the last move
            new = switch_tiles(tiles, width, x, y, 0)
            best_state, best_lc, change = compare_LCs(best_state, new, \
                best_lc, change, width)
            new_move = "H"
    if x > 0: # Possible to move right 
        if last_move != "H":    # Prevent reversing the last move
            new = switch_tiles(tiles, width, x, y, 1)
            best_state, best_lc, change = compare_LCs(best_state, new, \
                best_lc, change, width)
            new_move = "L"
    if y < width - 1: # Possible to move up 
        if last_move != "J":    # Prevent reversing the last move
            new = switch_tiles(tiles, width, x, y, 2)
            best_state, best_lc, change = compare_LCs(best_state, new, \
                best_lc, change, width)
            new_move = "K"
    if y > 0: # Possible to move down 
        if last_move != "K":    # Prevent reversing the last move
            new = switch_tiles(tiles, width, x, y, 3)
            best_state, best_lc, change = compare_LCs(best_state, new, \
                best_lc, change, width)
            new_move = "J"
    if not change:  # All next possible states have the same lc count
        best_state = new    # Set to most recent state explored
    last_move = new_move    # Set the move that was made
    return best_state, best_lc, change, last_move


def conflict_tiles(width: int, min_lc: int) -> Tuple[int, ...]:
    """
    Create a solvable shuffled puzzle of the given width with a minimum number
    of linear conflicts (ignoring Manhattan distance).
    Using hill climbing.
    >>> tiles = conflict_tiles(3, 5)
    >>> tiledriver.Heuristic._get_linear_conflicts(tiles, 3)
    5
    """
    best_lc: int = 0       # Number of linear conflicts of best state
    change: int = -1 
    max_plateau: int = 40
    k: int = 0
    last_move: str = ""

    while True:
        # Get a new random state
        if change < 0:  # Reached the peak, all next states are worse
            new_state = create_initial_state(width)
            best_lc = tiledriver.Heuristic._get_linear_conflicts(\
                new_state, width)   # New state, new LC Count
            last_move = ""      # Reset the last move to nothing
            k = 0               # Restarting, so reset plateau count
        elif change == 0:   # On a plateau, only go so far on the plateau
            k = k + 1
            if k > max_plateau:     # Reached end of plateau limit
                new_state = create_initial_state(width)

                best_lc = tiledriver.Heuristic._get_linear_conflicts(\
                    new_state, width)   # New state, new LC count
                last_move = ""   # Reset the last move to nothing
                k = 0               # Restarting, so reset plateau count
        else:   # Change was better, keep going on this path
            k = 0
        if best_lc >= min_lc:  # In case we get lucky with min linear conflicts
            return new_state
        # Otherwise, get the next possible states
        new_state, best_lc, change, last_move = \
            next_states(new_state, width, best_lc, last_move)
        if best_lc >= min_lc:   # Stop when we find the desired LC count
            break
    
    return new_state

def branch_next_states(tiles: Tuple[int, ...], width: int, \
last_move: str) -> Tuple[List[Tuple[int, ...]], List[str]]:
    """
    Get the set of new states based on which directions a tile can be moved.
    Keeps track of the last move made to that we are not back-tracking if 
    on a plateau.
    """
    pos: int = tiles.index(0)   # Index of the empty spot in list
    x: int = pos % width        # x position with respect to grid
    y: int = pos // width       # y position with respect to grid
    states: List[Tuple[int, ...]] = [] 
    moves: List[str] = []

    # Compare x and y to the width to see if it can go up, down, left or right
    if x < width - 1:       # Possible to move left
        if last_move != "L":    # Prevent reversing the last move
            new = switch_tiles(tiles, width, x, y, 0)
            states.append(new)
            moves.append("H")
    if x > 0: # Possible to move right 
        if last_move != "H":    # Prevent reversing the last move
            new = switch_tiles(tiles, width, x, y, 1)
            states.append(new)
            moves.append("L")
    if y < width - 1: # Possible to move up 
        if last_move != "J":    # Prevent reversing the last move
            new = switch_tiles(tiles, width, x, y, 2)
            states.append(new)
            moves.append("K")
    if y > 0: # Possible to move down 
        if last_move != "K":    # Prevent reversing the last move
            new = switch_tiles(tiles, width, x, y, 3)
            states.append(new)
            moves.append("J")
    return states, moves

def branch_states(base: List[Tuple[int, ...]], base_last_move: List[str], \
width: int, k: int, b: int) \
-> Tuple[List[Tuple[int, ...]], List[str], int]:
    
    """
    Create a solvable shuffled puzzle of the given width with an optimal
    solution length equal to or greater than the given minimum length.
    """
   
    i: int = 0
    j: int = 0
    count: int = 0
    branched_states: List[Tuple[int, ...]] = [None]*(k*b)
    branch_last_move: List[str] = [""]*(k*b) 
    states: List[Tuple[int, ...]] = [] 
    moves: List[str] = []

    # Take each base state and get the next two possible branches
    for i in range(k):
        states, moves = branch_next_states(base[i], width, base_last_move[i])
        j = 0
        while j < len(states) and j < b:
            branched_states[count] = states[j]
            branch_last_move[count] = moves[j]
            count = count + 1       # Number of states we are collecting
            j = j + 1
    return branched_states, branch_last_move, count

def check_heuristic(branches: List[Tuple[int, ...]], count: int, \
min_heuristic: int) -> Tuple[List[int], int, int]:
    """
    Check the heuristic of all the new branches to see which ones to keep
    """
    i: int = 0
    idx: int = 0
    h: List[int] = [0]*count
    path_length: int = 0

    for i in range(count):
        h[i] = tiledriver.Heuristic.get(branches[i])  # Get the heuristic
    for i in range(count):
        if h[i] > min_heuristic:
            path = tiledriver.solve_puzzle(branches[i])
            if len(path) > path_length:
                path_length = len(path)
                idx = i
    return h, path_length, idx


def choose_top_K_states(branches: List[Tuple[int, ...]], \
branch_last_move: List[str], h: List[int], k: int) \
-> Tuple[List[Tuple[int, ...]], List[str]]:
    """
    Choose K states out of the K*B branches from the last set of base states 
    """
    base_states: List[Tuple[int, ...]] = [] 
    base_last_move: List[str] = [] 
    # Choose K/2 top best heuristics
    sorted_H_idx = sorted(range(len(h)), key=lambda i: h[i])
    for _ in range(int(k/2)):
        base_states.append(branches[sorted_H_idx[-1]])
        base_last_move.append(branch_last_move[sorted_H_idx[-1]])
        sorted_H_idx.remove(sorted_H_idx[-1])

    # Choose K/2 random of the remaining branches
    random.shuffle(sorted_H_idx)
    for _ in range(int(k/2)):
        base_states.append(branches[sorted_H_idx[-1]])
        base_last_move.append(branch_last_move[sorted_H_idx[-1]])
        sorted_H_idx.remove(sorted_H_idx[-1])
    return base_states, base_last_move

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
    k: int = 0      # Active States (Keep this even!)
    b: int = 0      # Branching Factor
    min_heuristic: int = 0
    if width == 2:
        k = 4
        b = 2
        min_heuristic = 4 
    else: 
        k = 20 
        b = 3 
        min_heuristic = 24 
    count: int = 0
    base_states: List[Tuple[int, ...]] = [] 
    branches: List[Tuple[int, ...]] = []
    base_last_move: List[str] = [""]*k 
    branch_last_move: List[str] = [""]*(k*b) 
    h: List[int] = []
    path_length: int = 0

    # First grab the initial active states randomly
    for _ in range(k):
        base_states.append(create_initial_state(width))
    while True:
        # Branch them
        branches, branch_last_move, count = branch_states(base_states, \
            base_last_move, width, k, b)

        # Check the heuristic on all of them
        h, path_length, idx = check_heuristic(branches, count, min_heuristic)
    
        # Check if the path length is long enough
        if path_length >= min_len:
            return branches[idx] 

        # Take the top k/2 and then randomly choose the remaining states
        base_states, base_last_move = choose_top_K_states(branches, \
            branch_last_move, h, k)


def main() -> None:
    conflict_tiles(3, 10)
    #pass  # optional program test driver


if __name__ == "__main__":
    main()
