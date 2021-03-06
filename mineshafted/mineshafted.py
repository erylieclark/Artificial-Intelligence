# Name:         Erin Rylie Clark
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Mine Shafted
# Term:         Summer 2021

import itertools
from typing import Callable, Generator, List, Tuple, Set


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


def print_my_board(clues: List[int], width: int, length: int) \
-> List[List[int]]:
    """
    print the board that i have explored 
    """
    board: List[List[int]] = []
    for i in range(length):
        board.append(clues[i*width:i*width+width])
        print(board[i])
    print("Board: ", board)
    return board


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
domains: List[List[List[int]]], clues: List[int]) \
-> Tuple[List[List[List[int]]], List[int]]:
    """
    Open the specified new spot on the board and get the clue        
    """
    adj_idxs: List[int] = []
    clues[reveal_idx] = bm.move(reveal_idx)
    if not clues[reveal_idx]:
        # Get the adjacent indices
        adj_idxs = bm.get_adjacent(reveal_idx)
        adj_idxs = filter_adj_idxs(clues, adj_idxs)
        domains[reveal_idx] = get_domain(clues[reveal_idx], adj_idxs)
        domains[reveal_idx] = domains[reveal_idx][0:1]
    return domains, clues 


def get_domain(clue: int, adj_idxs: List[int]) -> List[List[int]]:
    """
    Get the domain of the spot just explored 
    """
    domain: List[List[int]] = []
    if clue:
        neg_pairs = list(itertools.combinations(range(len(adj_idxs)), clue))
    else:
        domain.append(adj_idxs)
        return domain

    neg_pairs = list(neg_pairs) 
    for _, pair in enumerate(neg_pairs):
        mine_comb = adj_idxs.copy()
        for j in range(clue):
            # Turn the mines negative
            mine_comb[pair[j]] = mine_comb[pair[j]]*-1
        domain.append(mine_comb)

    return domain   


def fill_new_domains(bm: BoardManager, domains: List[List[List[int]]], \
clues: List[int], new_reveals: List[int]) \
-> Tuple[List[List[List[int]]], List[int]]:
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
            undecided.append(i)

    return domains, undecided 


def get_new_arcs(undecided: List[int], arc_filter: int) -> List[List[int]]:
    """
    Get the arcs based on what was recently explored and what is undecided
    """
    i: int = 0
    queue: List[List[int]] = [[0]]*len(undecided)*(len(undecided)-1) 
    q = itertools.permutations(undecided, 2)
    for perm in list(q):
        queue[i] = list(perm)
        i += 1
    
    if arc_filter:
        i = 0
        while i < len(queue):
            if queue[i][1] != arc_filter:
                queue.pop(i)
            else:
                i += 1
    return queue


def select(lst: List, indices: List[int]) -> List:
    """
    Return a list containing only the indices indicated out of the given list
    """
    return list(lst[i] for i in indices)


def my_abs_list(lst: List[int]) -> List[int]:
    """
    Return absolute value of the list
    """
    for i in range(len(lst)):
        lst[i] = abs(lst[i])
    return lst


def shared_spaces(d1: List[List[int]], d2: List[List[int]]) \
-> Tuple[int, List[List[int]], List[List[int]]]:
    """
    Check if the arc actually has overlapping domains, return lists containing
    only the shared parts if yes
    """
    shared: int = False
    shared_spaces: List[int] = []
    j: int = 0
    keep_idx1: List[int] = []
    keep_idx2: List[int] = []
    d2_list: List[int] = my_abs_list(d2[0].copy())
    # Loop through d1 values and check if they are in d2
    for i, space in enumerate(d1[0]):
        if abs(space) in d2_list:
            shared_spaces.append(abs(space))
            keep_idx1.append(i)
            keep_idx2.append(d2_list.index(abs(space)))
            shared = True
            j += 1
    if shared:
        for i in range(len(d1)):
            d1[i] = select(d1[i], keep_idx1)
        for i in range(len(d2)):
            d2[i] = select(d2[i], keep_idx2)

    return shared, d1, d2


def compare_shared(d1: List[List[int]], d2: List[List[int]]) \
-> List[int]:
    """
    Check if any reductions can be made
    """
    s1: Set[int] = {0}
    s2: Set[int] = {0}
    idxs: List[int] = []
    for i in range(len(d1)):    # First reduce to just the subsets
        s1 = set(d1[i])
        for j in range(len(d2)):
            s2 = set(d2[j])
            if s1.issubset(s2):
                idxs.append(i)
                break
        
    return idxs

def reduce_domains(domains: List[List[List[int]]], \
queue: List[List[int]], undecided: List[int]) -> List[List[List[int]]]:
    """
    Recalc domains that contain an explored value
    """
    i: int = 0
    arc1: int = 0
    arc2: int = 0
    # Take out any list of the second value that does not contain a combination
    #   from the first value
    # Look in the second value to see if any of the numbers are consistently
    #   positive or negative, mark as either a mine or not a mine and add to
    #   reveal list if not a mine
    # If nothing comes of the change in domain, add it back to the queue
    
    while i < len(queue):
        arc1 = queue[i][0]
        arc2 = queue[i][1]
        
        shared, d1, d2 = shared_spaces(domains[arc1].copy(), \
            domains[arc2].copy())
        if shared: 
            keep_idxs = compare_shared(d1, d2)
            if len(keep_idxs) < len(domains[arc1]):
                # Reductions to be made
                domains[arc1] = select(domains[arc1], keep_idxs)
                # Add new arcs to queue
                queue.extend(get_new_arcs(undecided, arc1))
        queue.pop(i)

    return domains


def reduce_new_reveal_domains(domains: List[List[List[int]]], \
new_reveals: List[int]) -> Tuple[List[List[List[int]]], List[int]]:
    """
    Recalc domains that contain an explored value
    """
    undecided: List[int] = []
    for i in range(len(domains)):
        if len(domains[i]) > 1:
            for j in range(len(new_reveals)):
                shared, d1, d2 = shared_spaces(domains[i].copy(), \
                    [[new_reveals[j]]])
                if shared: 
                    keep_idxs = compare_shared(d1, d2)
                    if len(keep_idxs) < len(domains[i]):
                        domains[i] = select(domains[i], keep_idxs)
                        undecided.append(i)

    return domains, undecided


def compare_mines_domains(domains: List[List[List[int]]], clues: List[int], \
mines: List[int]) -> List[List[List[int]]]:
    """
    Gather more information using the known mine locations
    """
    for i in range(len(domains)):
        if len(domains[i]) > 1:
            shared, m, d = shared_spaces([mines].copy(), domains[i].copy())
            if shared: # and clues[i] == len(m):
                keep_idxs = compare_shared(d, m)
                domains[i] = select(domains[i], keep_idxs)

    return domains
    

def infer_safe_spots(domains: List[List[List[int]]], \
undecided: List[int]) -> Tuple[List[List[List[int]]], List[int], List[int]]:
    """
    Determine what spots are safe to explore next
    """
    new_reveals: List[int] = []
    mines: List[int] = []
    not_a_mine: int = False
    mine: int = 0
    spot: int = 0
    for i in range(len(undecided)):
        # Look at undecided spots
        if len(domains[undecided[i]]) == 1:
            # If only one left in domain, we know what is an isn't a mine
            for _, mine in enumerate(domains[undecided[i]][0]):
                if mine < 0:
                    mines.append(mine)
                else:
                    new_reveals.append(mine) 
        else:   # See if any consistent values
            d = domains[undecided[i]]
            spot = 0
            while spot < len(d[0]):
                for mine_set in range(len(d)):
                    if d[mine_set][spot] > 0:
                        not_a_mine = True
                    else:
                        not_a_mine = False
                        break
                if not_a_mine:
                    new_reveals.append(d[0][spot])
                    for mine_set in range(len(d)):
                        d[mine_set].pop(spot)
                    domains[undecided[i]] = d
                else:
                    spot += 1
    new_reveals = list(dict.fromkeys(new_reveals))
    mines = list(dict.fromkeys(mines))

    return domains, new_reveals, mines


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
    domains: List[List[List[int]]] = [[]]*board_size
    clues: List[int] = [-1]*board_size
    reveal_list: List[int] = [0]
    new_reveals: List[int] = []
    mines: List[int] = []
    arc_queue: List[List[int]] = [[]]
    # Make the first move by getting index 0, should be 0
    while True:
        new_reveals = []
        while len(reveal_list) > 0: # Open up the board as much as possible
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
        domains, undecided = reduce_new_reveal_domains(domains, new_reveals)
        domains, ud = fill_new_domains(bm, domains, clues, new_reveals)
        undecided.extend(ud)
        # Get arcs
        arc_queue = get_new_arcs(undecided, 0) 
        # Reduce Domains
        domains = reduce_domains(domains, arc_queue, undecided)
        domains = compare_mines_domains(domains, clues, mines)
        domains, new_reveals, m = infer_safe_spots(domains, undecided)
        mines.extend(m)
        if len(new_reveals) < 1:
            board = print_my_board(clues, board_width, board_length)
            return board
        else:
            reveal_list.extend(new_reveals)
        board = print_my_board(clues, board_width, board_length)
    return board


def main() -> None:  # optional driver
    board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    #board = [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, -1, 2, 1], \
    #    [1, 2, 1, 2, -1], [-1, 1, 0, 1, 1]]
    #board = [[0, 1, 1, 1], [1, 2, -1, 2], [1, -1, 3, -1], \
    #    [1, 1, 2, 1]]
    #board = [[0, 1, -1, 1], [1, 2, 1, 1], [-1, 1, 0, 0], \
    #    [1, 1, 0, 0], [0, 0, 0, 0]]
    #board = [[0, 1, 1, 1, 0, 0], [1, 2, -1, 1, 0, 0], \
    #    [-1, 2, 1, 2, 1, 1], [1, 1, 0, 1, -1, 1]]
    #board = [[0, 0, 0, 0], [1, 2, 2, 1], [1, -1, -1, 1], [1, 2, 2, 1]]
    
    bm = BoardManager(board)
    assert sweep_mines(bm) == board


if __name__ == "__main__":
    main()
