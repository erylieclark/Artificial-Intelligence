################################################################################
# Name:         Erin Rylie Clark
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Tile Driver I
# Term:         Summer 2021
################################################################################
import queue
from typing import List, Tuple


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
def next_states(tiles: Tuple[int, ...], x: int, y: int, width: int)\
 -> Tuple[int, ...]:
    """
    Get the new state based on which direction a tile will be sliding
    """
    states = {}
    # Compare x and y to the width to see if it can go up, down, left or right
    if x < width - 1: # Possible to move left
        print('Moving Left')
        new = list(tiles)
        print('Orig', new)
        new[x + y*width] = new[x + y*width + 1]
        new[x + y*width + 1] = 0
        print('New', new)
        states.update({"H": [new]})
    if x > 0: # Possible to move right 
        print('Moving Right')
        new = list(tiles)
        print('Orig', new)
        new[x + y*width] = new[x + y*width - 1]
        new[x + y*width - 1] = 0
        print('New', new)
        states.update({"L": [new]})
    if y < width - 1: # Possible to move up 
        print('Moving Up')
        new = list(tiles)
        print('Orig', new)
        new[x + y*width] = new[x + y*width + width]
        new[x + y*width + width] = 0
        print('New', new)
        states.update({"K": [new]})
    if y > 0: # Possible to move down 
        print('Moving Down')
        new = list(tiles)
        print('Orig', new)
        new[x + y*width] = new[x + y*width - width]
        new[x + y*width - width] = 0
        print('New', new)
        states.update({"L": [new]})
        
################################################################################
def frontier_states(tiles: Tuple[int, ...]) -> dict:
    """
    Determine what states can be next given an arbitrary current state.
    """
    states = {}
    print('Getting Frontier States')
    width = int(len(tiles) ** 0.5)
    print('width= ',width)
    pos0 = tiles.index(0)
    print('pos0= ',pos0)
    # Get the x and y position of the 0
    x0 = pos0 % width
    y0 = pos0 // width
    print('x0= ',x0)
    print('y0= ',y0)
    # Compare x and y to the width to see if it can go up, down, left or right
    '''
    if x0 < width - 1: # Possible to move left
        new = new_state(tiles, x0, y0, width, "left") 
        states.update({"H": [new]})
    if x0 > 0: # Possible to move right 
        new = new_state(tiles, x0, y0, width, "right") 
        states.update({"L": [new]})
    if y0 < width - 1: # Possible to move up 
        new = new_state(tiles, x0, y0, width, "up") 
        states.update({"K": [new]})
    if y0 > 0: # Possible to move down 
        new = new_state(tiles, x0, y0, width, "down") 
        states.update({"L": [new]})
    '''
    states = next_states(tiles, x0, y0, width)
    # Find a way to add a switched version of the Tuple to a dictionary 
    
    return states

################################################################################
def solve_puzzle(tiles: Tuple[int, ...]) -> str:
    """
    Return a string (containing characters "H", "J", "K", "L") representing the
    optimal number of moves to solve the given puzzle.
    """
    frontiers = frontier_states(tiles)
    
    h = Heuristic.get(tiles)
    print('Heuristic:')
    print(h)
    return 'Some String'



################################################################################
""" 
    1. Given the initial state, determine what the next possible states are.
    2. Get the Heuristic for each of those states and add states to the queue.
    3. Check if one of those states is the end state?
    4. Add dictionary entry with the string of moves to get to that state.
"""
def main() -> None:
    init_state = 3,7,1,4,0,2,6,8,5 
    # init_state = 0,2,3,1
    path = solve_puzzle(init_state)
    print(path)
    pass  # optional program test driver


if __name__ == "__main__":
    main()
