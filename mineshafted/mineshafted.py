# Name:         Erin Rylie Clark
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Mine Shafted
# Term:         Summer 2021

import itertools
from typing import Callable, Generator, List, Tuple


class BoardManager:  # do not modify

    def __init__(self, board: List[List[int]]):
        """
        An instance of BoardManager has two attributes.

            size: A 2-tuple containing the number of rows and columns,
                  respectively, in the game board.
            move: A callable that takes an integer as its only argument to be
                  used as the index to explore on the board. If the value at
                  that index is a clue (non-mine), this clue is returned;
                  otherwise, an error is raised.

        This constructor should only be called once per game.

        >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
        >>> bm = BoardManager(board)
        >>> bm.size
        (4, 3)
        >>> bm.move(4)
        2
        >>> bm.move(5)
        Traceback (most recent call last):
        ...
        RuntimeError
        """
        self.size: Tuple[int, int] = (len(board), len(board[0]))
        it: Generator[int, int, None] = BoardManager._move(board, self.size[1])
        next(it)
        self.move: Callable[[int], int] = it.send

    def get_adjacent(self, index: int) -> List[int]:
        """
        Return a list of indices adjacent (including diagonally) to the given
        index. All adjacent indices are returned, regardless of whether or not
        they have been explored.

        >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
        >>> bm = BoardManager(board)
        >>> bm.get_adjacent(3)
        [0, 1, 4, 6, 7]
        """
        row, col = index // self.size[1], index % self.size[1]
        return [i * self.size[1] + j
                for i in range(max(row - 1, 0), min(row + 2, self.size[0]))
                for j in range(max(col - 1, 0), min(col + 2, self.size[1]))
                if index != i * self.size[1] + j]

    @staticmethod
    def _move(board: List[List[int]], width: int) -> Generator[int, int, None]:
        """
        A generator that may be sent integers (indices to explore on the board)
        and yields integers (clues for the explored indices).

        Do not call this method directly; instead, call the |move| instance
        attribute, which sends its index argument to this generator.
        """
        index = (yield 0)
        while True:
            clue = board[index // width][index % width]
            if clue == -1:
                raise RuntimeError
            index = (yield clue)


def print_my_board(clues: List[int], width: int, length: int):
    """
    print the board that i have explored 
    """
    for i in range(length):
        print(clues[i*width:i*width+width])


def print_domains(domains: List[List[int]], clues: List[int]):
    """
    print the domains of the unexplored spots 
    """
    print("----------------------- Domains --------------------")
    for i in range(len(clues)):
        if clues[i] == -1:
            print("IDX: ", domains[i])


def filter_adj_idxs(clues: List[int], adj_idxs: List[int]) -> List[int]:
    """
    Remove explored spots from the adjacent spots list 
    """
    i: int = 0
    while i < len(adj_idxs):
        if clues[adj_idxs[i]] != -1:
            adj_idxs.pop(i)
        else:
            i += 1

    return adj_idxs


def explore_new_spot(bm: BoardManager, reveal_idx: int, \
domains: List[List[int]], clues: List[int]) \
-> Tuple[List[List[int]], List[int]]:
    """
    Open the specified new spot on the board and get the clue        
    """
    adj_idxs: List[int] = []
    clues[reveal_idx] = bm.move(reveal_idx)
    print("Index: ", reveal_idx)
    #print("Clue: ", clues[reveal_idx])
    if not clues[reveal_idx]:
        # Get the adjacent indices
        adj_idxs = bm.get_adjacent(reveal_idx)
        #print("Adjacent Indexes: ", adj_idxs)
        adj_idxs = filter_adj_idxs(clues, adj_idxs)
        #print("Filtered Adjacent Indexes: ", adj_idxs)
        domains[reveal_idx] = get_domain(clues[reveal_idx], adj_idxs)
        domains[reveal_idx] = domains[reveal_idx][0:1]
    print("Domain: ", domains[reveal_idx])
    return domains, clues 


def get_domain(clue: int, adj_idxs: List[int]) -> List[int]:
    """
    Get the domain of the spot just explored 
    """
    domain: List[List[int]] = [[]]*len(adj_idxs) 
    for i in range(-1, len(adj_idxs)-1):
        mine_idx = adj_idxs.copy()
        for j in range(clue):
            mine_idx[i+j] = mine_idx[i+j]*-1     
        domain[i+1] = mine_idx

    return domain   


def fill_new_domains(bm: BoardManager, domains: List[List[int]], \
clues: List[int], new_reveals: List[int]) -> Tuple[List[List[int]], List[int]]:
    """
    Recalc domains that contain an explored value
    """
    # Pass in list of new domains that need to be cleaned up
    undecided: List[int] = []
    # One method: find all domains that have more than 1 in the list, then
    #   recalc adjacent vals, filter, and get domains
    for i in range(len(domains)):
        if len(domains[i]) > 1:
            undecided.append(i)
            
        elif i in new_reveals:
            adj_idxs = bm.get_adjacent(i)
            adj_idxs = filter_adj_idxs(clues, adj_idxs)
            domains[i] = get_domain(clues[i], adj_idxs) 
            print("Domain for index", i)
            print("     ", domains[i])
            undecided.append(i)

    return domains, undecided 
     


def get_new_arcs(undecided: List[int]) -> List[List[int]]:
#queue: List[List[int]]) 
    """
    Get the arcs based on what was recently explored and what is undecided
    """
    i: int = 0
    queue: List[List[int]] = [[]]*len(undecided)*(len(undecided)-1) 
    q = itertools.permutations(undecided, 2)
    for perm in list(q):
        queue[i] = perm
        i += 1

    print("Queue: ", queue)

    return queue


def reduce_domains(domains: List[List[int]], queue: List[List[int]]) \
-> Tuple[List[List[int]], List[int]]:
    """
    Recalc domains that contain an explored value
    """
    reveal_list: List[int] = []
    # Take out any list of the second value that does not contain a combination
    #   from the first value
    # Look in the second value to see if any of the numbers are consistently
    #   positive or negative, mark as either a mine or not a mine and add to
    #   reveal list if not a mine
    # If nothing comes of the change in domain, add it back to the queue
    


    return domains, reveal_list

def sweep_mines(bm: BoardManager) -> List[List[int]]:
    """
    Given a BoardManager (bm) instance, return a solved board (represented as a
    2D list) by repeatedly calling bm.move(index) until all safe indices have
    been explored. If at any time a move is attempted on a non-safe index, the
    BoardManager will raise an error; this error signifies the end of the game
    and should not attempt to be caught.

    >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    >>> bm = BoardManager(board)
    >>> sweep_mines(bm)
    [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    """
    # Figure out how big the board is
    board_length, board_width = bm.size
    board_size: int = board_width*board_length 
    # Create a list of lists to keep track of explored states
    domains: List[List[int]] = [[]]*board_size
    clues: List[int] = [-1]*board_size
    reveal_list: List[int] = [0]
    new_reveals: List[int] = []
    arc_queue: List[List[int]] = [[]]
    # Make the first move by getting index 0, should be 0
    while True:
        new_reveals = []
        while len(reveal_list): # Open up the board as much as possible
            domains, clues = explore_new_spot(bm, reveal_list[0], \
                domains, clues)
            if not clues[reveal_list[0]]: # If the clue was zero
                # Add all of the adjacent spots to the board
                reveal_list.extend(domains[reveal_list[0]][0])
                # Remove duplicates
                reveal_list = list(dict.fromkeys(reveal_list))
            else:
                # Keep track of spots that have mines next to them
                new_reveals.append(reveal_list[0])
            reveal_list.pop(0) # Remove revealed index from list
        print("New Reveals: ", new_reveals)
        domains, undecided = fill_new_domains(bm, domains, clues, new_reveals)
        print("Undecided: ", undecided)
        # Get arcs
        arc_queue = get_new_arcs(undecided) 
        # Reduce Domains
        domains, reveal_list = reduce_domains(domains, arc_queue)
        print_my_board(clues, board_width, board_length)
        #print_domains(domains, clues)
        break


def main() -> None:  # optional driver
    board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    bm = BoardManager(board)
    assert sweep_mines(bm) == board


if __name__ == "__main__":
    main()
