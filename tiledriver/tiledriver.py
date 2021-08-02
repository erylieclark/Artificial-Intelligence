# Name:         Erin Rylie Clark
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Tile Driver I
# Term:         Summer 2021

import queue
from typing import List, Tuple, Dict

################################################################################
class TileState:
    def __init__(self, heuristic: int, path: str, cost: int):
        self.heuristic: int = heuristic
        self.path: str = path
        self.cost: int = cost

    def new_path(self, path: str):
        self.path: str = path

    def new_cost(self, cost: int):
        self.cost: int = cost 
    

################################################################################
class Heuristic:

    @staticmethod
    def get(tiles: Tuple[int, ...]) -> int:
        """
        Return the estimated distance to the goal using Manhattan distance
        and linear conflicts.

        Only this static method should be called during a search; all other
        methods in this class should be considered private.

        >>> Heuristic.get((0, 1, 2, 3))
        0
        >>> Heuristic.get((3, 2, 1, 0))
        6
        """
        width = int(len(tiles) ** 0.5)
        return (Heuristic._get_manhattan_distance(tiles, width)
                + Heuristic._get_linear_conflicts(tiles, width))

    @staticmethod
    def _get_manhattan_distance(tiles: Tuple[int, ...], width: int) -> int:
        """
        Return the Manhattan distance of the given tiles, which represents
        how many moves is tile is away from its goal position.
        """
        distance = 0
        for i in range(len(tiles)):
            if tiles[i] != 0:
                row_dist = abs(i // width - tiles[i] // width)
                col_dist = abs(i % width - tiles[i] % width)
                distance += row_dist + col_dist
        return distance

    @staticmethod
    def _get_linear_conflicts(tiles: Tuple[int, ...], width: int) -> int:
        """
        Return the number of linear conflicts in the tiles, which represents
        the minimum number of tiles in each row and column that must leave and
        re-enter that row or column in order for the puzzle to be solved.
        """
        conflicts = 0
        rows = [[] for i in range(width)]
        cols = [[] for i in range(width)]
        for i in range(len(tiles)):
            if tiles[i] != 0:
                if i // width == tiles[i] // width:
                    rows[i // width].append(tiles[i])
                if i % width == tiles[i] % width:
                    cols[i % width].append(tiles[i])
        for i in range(width):
            conflicts += Heuristic._count_conflicts(rows[i])
            conflicts += Heuristic._count_conflicts(cols[i])
        return conflicts * 2

    @staticmethod
    def _count_conflicts(ints: List[int]) -> int:
        """
        Return the minimum number of tiles that must be removed from the given
        list in order for the list to be sorted.
        """
        if Heuristic._is_sorted(ints):
            return 0
        lowest = None
        for i in range(len(ints)):
            conflicts = Heuristic._count_conflicts(ints[:i] + ints[i + 1:])
            if lowest is None or conflicts < lowest:
                lowest = conflicts
        return 1 + lowest

    @staticmethod
    def _is_sorted(ints: List[int]) -> bool:
        """Return True if the given list is sorted and False otherwise."""
        for i in range(len(ints) - 1):
            if ints[i] > ints[i + 1]:
                return False
        return True


################################################################################
def next_states(tiles: Tuple[int, ...], x: int, y: int, width: int, \
last_char: str) -> dict:
    """
    Get the set of new states based on which directions a tile can be moved.
    Return a dictionary containing each of the new states.
    """
    states = {}
    # Compare x and y to the width to see if it can go up, down, left or right
    if x < width - 1: # Possible to move left
        if last_char != "L":    # Prevent reversing the last move
            new = list(tiles)   # Turn Tuple into a list to modify
            new[x + y*width] = new[x + y*width + 1] # Move # to 0 spot
            new[x + y*width + 1] = 0        # Replace # with empty spot
            states.update({"H": tuple(new)})# Add to dictionary
    if x > 0: # Possible to move right 
        if last_char != "H":    # Prevent reversing the last move
            new = list(tiles)   # Turn Tuple into a list to modify
            new[x + y*width] = new[x + y*width - 1] # Move # to 0 spot
            new[x + y*width - 1] = 0        # Replace # with empty spot
            states.update({"L": tuple(new)})# Add to dictionary
    if y < width - 1: # Possible to move up 
        if last_char != "J":    # Prevent reversing the last move
            new = list(tiles)   # Turn Tuple into a list to modify
            new[x + y*width] = new[x + y*width + width] # Move # to 0 spot
            new[x + y*width + width] = 0    # Replace # with empty spot
            states.update({"K": tuple(new)})# Add to dictionary
    if y > 0: # Possible to move down 
        if last_char != "K":    # Prevent reversing the last move
            new = list(tiles)   # Turn Tuple into a list to modify
            new[x + y*width] = new[x + y*width - width] # Move # to 0 spot
            new[x + y*width - width] = 0    # Replace # with empty spot
            states.update({"J": tuple(new)})# Add to dictionary
    return states
        
################################################################################
def check_frontier(frontiers: dict, new_states: dict, cur_state: TileState, \
tiles: Tuple[int, ...], q: queue.PriorityQueue) -> \
Tuple[dict, queue.PriorityQueue]: 
    """
    Determine if the next possible states from the state at the top of the
    priority queue are already on the frontier. If they are, check if the new
    path to that state is shorter than the existing one and replace it if it is
    shorter. If they states are not already on the frontier, add them to the
    frontier and to the queue, and return both.
    """
    frontiers.update({tiles: cur_state})    # Chosen state based on queue
    directions = ("H", "J", "K", "L")       # Add these to end of path

    # Check to see if the state is already in the dictionary
    for direction in enumerate(directions):
        key = direction[1]                  # Key is the char, H/J/K/L
        if key in new_states.keys():        # Check if direction is applicable
            cost = cur_state.cost + 1       # Each move costs 1 more
            path = cur_state.path + key     # Add it to the potential path 
            if new_states[key] in frontiers.keys(): 
                # Check if state has been on frontier 
                existing_state = frontiers[new_states[key]]
                    # existing_state == object of the existing state
                if cost < existing_state.cost:      # Compare cost to get there
                    # Enter here if the new path to this state costs less than
                    #   the path that was already on the frontier
                    existing_state.new_cost(cost)   # Update the cost
                    existing_state.new_path(path)   # Update the path
            else:   # Not on frontier, so add it
                h = Heuristic.get(new_states[key])  # Get the heuristic
                obj = TileState(h, path, cost)      # Create the dict object
                frontiers.update({new_states[key]: obj})    # Add to frontier
                q.put((h + cost, new_states[key]))          # Add to queue
    return frontiers, q

################################################################################
def solve_puzzle(tiles: Tuple[int, ...]) -> str:
    """
    Return a string (containing characters "H", "J", "K", "L") representing the
    optimal number of moves to solve the given puzzle.
    """
    new_states: Dict[Tuple[int, ...], TileState] = {}
    frontiers: Dict[Tuple[int, ...], TileState] = {}
    q: queue.PriorityQueue = queue.PriorityQueue()
    width: int = int(len(tiles) ** 0.5)     # Width of puzzle

    # Add first state into the dictionary
    h: int = Heuristic.get(tiles)           # First get the heuristic
    if not h:
        return ""                           # Already at final state
    obj: TileState = TileState(h, "", 0)    # Cost is 0 for first state, no path
    cur_state: TileState = obj
    last_char: str = "X"

    # Cycle through until we find the final state
    while True:
        pos0: int = tiles.index(0)          # Index of the empty spot in list
        x0: int = pos0 % width              # x position with respect to grid
        y0: int = pos0 // width             # y position with respect to grid
        # Find what the next possible states are and return in a dictionary
        new_states = next_states(tiles, x0, y0, width, last_char)
        # Get which state to explore next and updated list of frontiers 
        frontiers, q = check_frontier(frontiers, new_states, \
            cur_state, tiles, q)
        if not q.empty():       # Check that the queue is not empty first
            _, tiles = q.get()  # Get the least costly frontier state
        else:
            if cur_state.heuristic:
                # Unless heuristic is zero, out of frontiers to check
                print("No Possible Soluion")
            break
        cur_state = frontiers[tiles]    # Update the new current state
        last_char = cur_state.path[-1]  # Last character to prevent reversing
        if not cur_state.heuristic:     # Stop when heuristic of cur_state is 0
            break
    return str(cur_state.path)          # Return the path



################################################################################
 
def main() -> None:
    # init_state = 0, 3, 6, 5, 4, 7, 2, 1, 8 
    # init_state = 3, 7, 1, 4, 0, 2, 6, 8, 5 
    # init_state = 2, 1, 0, 3 
    # init_state = 6, 7, 8, 3, 0, 5, 1, 2, 4
    # init_state = 8, 2, 0, 5, 4, 3, 7, 1, 6
    init_state = 0, 1, 2, 3
    path = solve_puzzle(init_state)
    print(path)
    #pass  # optional program test driver


if __name__ == "__main__":
    main()
