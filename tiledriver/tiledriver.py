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
    Get the new state based on which direction a tile will be sliding
    """
    print("\nGetting next states...")
    print("    Last Character: ", last_char)
    states = {}
    # Compare x and y to the width to see if it can go up, down, left or right
    if x < width - 1: # Possible to move left
        if last_char != "L":     # Prevent reversing the last move
            new = list(tiles) 
            new[x + y*width] = new[x + y*width + 1]
            new[x + y*width + 1] = 0
            states.update({"H": tuple(new)})
    if x > 0: # Possible to move right 
        if last_char != "H":     # Prevent reversing the last move
            new = list(tiles) 
            new[x + y*width] = new[x + y*width - 1]
            new[x + y*width - 1] = 0
            states.update({"L": tuple(new)})
    if y < width - 1: # Possible to move up 
        if last_char != "J":
            new = list(tiles) 
            new[x + y*width] = new[x + y*width + width]
            new[x + y*width + width] = 0
            states.update({"K": tuple(new)})
    if y > 0: # Possible to move down 
        if last_char != "K":
            new = list(tiles) 
            new[x + y*width] = new[x + y*width - width]
            new[x + y*width - width] = 0
            states.update({"J": tuple(new)})
    return states
        
################################################################################
def check_frontier(frontiers: dict, new_states: dict, cur_state: TileState, \
tiles: Tuple[int, ...], q: queue.PriorityQueue) -> \
Tuple[dict, queue.PriorityQueue]: 
#-> Tuple[dict, tuple[int,...],...]:
    """
    Writing something to make the program happy
    """
    frontiers.update({tiles: cur_state})
    directions = ("H", "J", "K", "L")
    # Check to see if the state is already in the dictionary
    for direction in enumerate(directions):
        key = direction[1] 
        if key in new_states.keys():      # Check if direction is applicable
            print("\n")
            print("Possible Key: ", key)
            print("    Associated State: ", new_states[key])
            cost = cur_state.cost + 1 
            print("    Cost: ", cost)
            path = cur_state.path + key 
            print("    Path: ", path)
            if new_states[key] in frontiers.keys(): 
                print("Already on frontier...")
                # Check if state is on frontier 
                # Compare the cost to get there
                existing_state = frontiers[new_states[key]]
                    # Object of the existing state
                print("    Cost of Existing State: ", existing_state.cost)
                print("    Cost of New State: ", cost)
                if cost < existing_state.cost:
                    print("... Choose New State")
                    existing_state.new_cost(cost)   # Update the cost
                    existing_state.new_path(path)   # Update the path
                else:
                    print("... Choose Existing State")
                    #frontiers[new_states[i]]  
            else:   # Not on frontier, add it
                h = Heuristic.get(new_states[key])    # Get the heuristic
                print("    Heuristic: ", h)
                obj = TileState(h, path, cost) 
                #print("Updating Frontiers..")
                frontiers.update({new_states[key]: obj})
                q.put((h + cost, new_states[key]))    # Add to queue
                print("    Total Cost: ", h + cost)
            # Could I accidentally circle back to the same state?
            # Need I keep track of all explored states or only frontiers?
    # Now check which path should be the new path
    return frontiers, q

################################################################################
#def explore_next(frontiers: dict, q: object) -> object:
    
#    return 

################################################################################
def solve_puzzle(tiles: Tuple[int, ...]) -> str:
    """
    Return a string (containing characters "H", "J", "K", "L") representing the
    optimal number of moves to solve the given puzzle.
    """
    # Should check to make sure that length of tiles is appropriate
    # Maybe check to make sure there are 1 of each number and the right numbs
    # Check for states that may not be able to be solved
    # Check that we are not repeating moves
    # Check that the queue is not empty
    # Think of as many edge cases as you can

    state_dict: Dict[Tuple[int, ...], TileState] = {}
    new_states: Dict[Tuple[int, ...], TileState] = {}
    frontiers: Dict[Tuple[int, ...], TileState] = {}
    q: queue.PriorityQueue = queue.PriorityQueue()
    width: int = int(len(tiles) ** 0.5)      # Width of puzzle

    # Add first state into the dictionary
    h: int = Heuristic.get(tiles)          # First get the heuristic
    obj: TileState = TileState(h, "", 0)   # Cost is 0 for first state, no path
    state_dict.update({tiles:obj})      # Add to dictionary
    cur_state: TileState = obj
    last_char: str = "X"
    # Cycle through until we find the final state
    while True:
        pos0: int = tiles.index(0)           # Index of the empty spot in list
        x0: int = pos0 % width               # x position with respect to grid
        y0: int = pos0 // width              # y position with respect to grid
        # Find what the next possible states are and return in a dictionary
        new_states = next_states(tiles, x0, y0, width, last_char)
        # Get which state to explore next and updated list of frontiers 
        # cur_state = state_dict[tiles]
        frontiers, q = check_frontier(frontiers, new_states, \
            cur_state, tiles, q)
        # Check that the queue is not empty!
        #while q:
        #print(q.get())
        if not q.empty():
            _, tiles = q.get()
            print("\nExplore next: ", tiles, "\n")
        else:
            print("Queue is empty")
            if cur_state.heuristic:
                print("No Possible Soluion")
            break
        #print("Pri: ", pri)
        #print("Next state: ", tiles) 
        # Remove explored state from frontiers
        print("Path: ", (frontiers[tiles]).path) 
        print("State: ", tiles)
        cur_state = frontiers[tiles] 
        last_char = cur_state.path[-1]
        #print("Heuristic: ", cur_state.heuristic)
        #print(cur_state.heuristic)
        if not cur_state.heuristic:
            break
    return str(cur_state.path)



################################################################################
 
# 1. Given the initial state, determine what the next possible states are.
# 2. Get the Heuristic for each of those states and add states to the queue.
# 3. Check if one of those states is the end state?
# 4. Add dictionary entry with the string of moves to get to that state.

def main() -> None:
    # init_state = 3, 7, 1, 4, 0, 2, 6, 8, 5 
    init_state = 2, 1, 0, 3 
    path = solve_puzzle(init_state)
    print(path)
    #pass  # optional program test driver


if __name__ == "__main__":
    main()
