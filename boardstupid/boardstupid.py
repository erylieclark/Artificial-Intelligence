# Name:         Erin Rylie Clark
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Board Stupid
# Term:         Summer 2021

import math
import random
from typing import Callable, Generator, Optional, Tuple


class GameState:

    def __init__(self, board: Tuple[Tuple[Optional[int], ...], ...],
                 player: int) -> None:
        """
        An instance of GameState has the following attributes.

            player: Set as either 1 (MAX) or -1 (MIN).
            moves: A tuple of integers representing empty indices of the board.
            selected: The index that the current player believes to be their
                      optimal move; defaults to -1.
            util: The utility of the board; either 1 (MAX wins), -1 (MIN wins),
                  0 (tie game), or None (non-terminal game state).
            traverse: A callable that takes an integer as its only argument to
                      be used as the index to apply a move on the board,
                      returning a new GameState object with this move applied.
                      This callable provides a means to traverse the game tree
                      without modifying parent states.
            display: A string representation of the board, which should only be
                     used for debugging and not parsed for strategy.

        In addition, instances of GameState may be stored in hashed
        collections, such as sets or dictionaries.

        >>> board = ((   0,    0,    0,    0,   \
                         0,    0, None, None,   \
                         0, None,    0, None,   \
                         0, None, None,    0),) \
                    + ((None,) * 16,) * 3

        >>> state = GameState(board, 1)
        >>> state.util
        None
        >>> state.player
        1
        >>> state.moves
        (0, 1, 2, 3, 4, 5, 8, 10, 12, 15)
        >>> state = state.traverse(0)
        >>> state.player
        -1
        >>> state.moves
        (1, 2, 3, 4, 5, 8, 10, 12, 15)
        >>> state = state.traverse(5)
        >>> state.player
        1
        >>> state.moves
        (1, 2, 3, 4, 8, 10, 12, 15)
        >>> state = state.traverse(1)
        >>> state.player
        -1
        >>> state.moves
        (2, 3, 4, 8, 10, 12, 15)
        >>> state = state.traverse(10)
        >>> state.player
        1
        >>> state.moves
        (2, 3, 4, 8, 12, 15)
        >>> state = state.traverse(2)
        >>> state.player
        -1
        >>> state.moves
        (3, 4, 8, 12, 15)
        >>> state = state.traverse(15)
        >>> state.player
        1
        >>> state.moves
        (3, 4, 8, 12)
        >>> state = state.traverse(3)
        >>> state.util
        1
        """
        self.player: int = player
        self.moves: Tuple[int] = GameState._get_moves(board, len(board))
        self.selected: int = -1
        self.util: Optional[int] = GameState._get_utility(board, len(board))
        self.traverse: Callable[[int], GameState] = \
            lambda index: GameState._traverse(board, len(board), player, index)
        self.display: str = GameState._to_string(board, len(board))
        self.keys: Tuple[int, ...] = tuple(hash(single) for single in board)

    def __eq__(self, other: "GameState") -> bool:
        return self.keys == other.keys

    def __hash__(self) -> int:
        return hash(self.keys)

    @staticmethod
    def _traverse(board: Tuple[Tuple[Optional[int], ...], ...],
                  width: int, player: int, index: int) -> "GameState":
        """
        Return a GameState instance in which the board is updated at the given
        index by the current player.

        Do not call this method directly; instead, call the |traverse| instance
        attribute, which only requires an index as an argument.
        """
        i, j = index // width ** 2, index % width ** 2
        single = board[i][:j] + (player,) + board[i][j + 1:]
        return GameState(board[:i] + (single,) + board[i + 1:], -player)

    @staticmethod
    def _get_moves(board: Tuple[Tuple[Optional[int], ...], ...],
                   width: int) -> Tuple[int]:
        """
        Return a tuple of the unoccupied indices remaining on the board.
        """
        return tuple(j + i * width ** 2 for i, single in enumerate(board)
                     for j, square in enumerate(single) if square == 0)

    @staticmethod
    def _get_utility(board: Tuple[Tuple[Optional[int], ...], ...],
                     width: int) -> Optional[int]:
        """
        Return the utility of the board; either 1 (MAX wins), -1 (MIN wins),
        0 (tie game), or None (non-terminal game state).
        """
        for line in GameState._iter_lines(board, width):
            if line == (1,) * width:
                return 1
            if line == (-1,) * width:
                return -1
        return 0 if len(GameState._get_moves(board, width)) == 0 else None

    @staticmethod
    def _iter_lines(board: Tuple[Tuple[Optional[int], ...], ...],
                    width: int) -> Generator[Tuple[int], None, None]:
        """
        Iterate over all groups of indices that represent a winning condition.
        X lines are row-wise, Y lines are column-wise, and Z lines go through
        all single boards; combinations of these axes refer to the direction
        of the line in 2D or 3D space.
        """
        for single in board:
            # x lines (2D rows)
            for i in range(0, len(single), width):
                yield single[i:i + width]
            # y lines (2D columns)
            for i in range(width):
                yield single[i::width]
            # xy lines (2D diagonals)
            yield single[::width + 1]
            yield single[width - 1:len(single) - 1:width - 1]
        # z lines
        for i in range(width ** 2):
            yield tuple(single[i] for single in board)
        for j in range(width):
            # xz lines
            yield tuple(board[i][j * width + i] for i in range(len(board)))
            yield tuple(board[i][j * width + width - 1 - i]
                        for i in range(len(board)))
            # yz lines
            yield tuple(board[i][j + i * width] for i in range(len(board)))
            yield tuple(board[i][-j - 1 - i * width]
                        for i in range(len(board)))
        # xyz lines
        yield tuple(board[i][i * width + i] for i in range(len(board)))
        yield tuple(board[i][i * width + width - 1 - i]
                    for i in range(len(board)))
        yield tuple(board[i][width ** 2 - width * (i + 1) + i]
                    for i in range(len(board)))
        yield tuple(board[i][width ** 2 - (i * width) - i - 1]
                    for i in range(len(board)))

    @staticmethod
    def _to_string(board: Tuple[Tuple[Optional[int], ...], ...],
                   width: int) -> str:
        """
        Return a string representation of the game board, in which integers
        represent the indices of empty spaces and the characters "X" and "O"
        represent previous move selections for MAX and MIN, repsectively.
        """
        display = "\n"
        for i in range(width):
            for j in range(width):
                line = board[j][i * width:i * width + width]
                start = j * width ** 2 + i * width
                for k, space in enumerate(line):
                    if space == 0:
                        space = start + k
                    else:
                        space = ("X" if space == 1
                                 else "O" if space == -1
                                 else "-")
                    display += "{0:>4}".format(space)
                display += " " * width
            display += "\n"
        return display

class State:
    def __init__(self, idx: int, ucb: float, state):
        self.idx = idx
        self.ucb = ucb
        self.move_ucbs = []
        self.moves = []
        self.wins = 0
        self.attempts = 0
        self.board = state

    def add_move(self, move, ucb):
        self.moves.append(move)
        self.move_ucbs.append(ucb)

    def winner(self):
        self.wins += 1
        self.attempts += 1

    def loser(self):
        self.attempts += 1


def expand(child: State) -> State:
    """
    Get the moves that the child can make and select one at random
    """
    c: float = 2 ** 0.5 # Start with sqrt(2)

    state = child.board
    #print("Child Moves: ", state.moves)
    for _, m in enumerate(state.moves):
        board = state.traverse(m)
        # TODO: Check if the board is solved
        new_move = State(m, c, board)   # Add index of the move and C as the UCB
        child.add_move(new_move, c)      # Add the object to the list of moves
    return child


def simulate(state: GameState) -> int:
    """
    Get the moves that the child can make and select one at random
    """
    moves = list(state.moves)
    for _ in range(len(state.moves)):
        move = random.choice(moves)
        move_idx = moves.index(move)
        moves.pop(move_idx)
        state = state.traverse(move)
    #print("    Winner: ", state.util)
    #print("         New Board: ", state.display)
    return state.util
   

def update_ucbs(parent: State, c: float = 2 ** 0.5) -> None:
    for i, child in enumerate(parent.moves): 
        if child.attempts > 0:
            fraction = (math.log(parent.attempts)/child.attempts) ** 0.5
            child.ucb = child.wins/child.attempts + c * fraction
            parent.move_ucbs[i] = child.ucb


def update_selected(move: State, win_ratio: float)\
-> Tuple[int, float]:
    idx: int = -1
    ratio: float = move.wins/move.attempts
    if ratio > win_ratio:
        idx = move.idx
        win_ratio = ratio
    else:
        idx = None
    return idx, win_ratio


def run_each_move(root: State, state: GameState) -> State:
    """
    Run each initial move a number of times to get a better idea
    of which ones will work best
    """
    for m, base_move in enumerate(root.moves):
        for _ in range(10):
            base_move = expand(base_move)
            _, outcome, _ = mcts(base_move, state, False, 0)
            if outcome == state.player:
                base_move.winner()
            else:
                base_move.loser()
            root.attempts += 1
        root.moves[m] = base_move
    update_ucbs(root)
    return root 
    

def mcts(parent: State, state: GameState, root: int, win_ratio: float)\
 -> Tuple[int, int, float]:
    """
    Perform Monte Carlo Tree Search
    """
    new_winner: int = None
    # Find the maximum UCB value to explore
    idx = parent.move_ucbs.index(max(parent.move_ucbs))
    # Select the child with the highest UCB value
    child = parent.moves[idx]
    # Check if the child has moves expanded already
    if child.attempts == 0 or len(child.moves) == 0: # Expand if no moves
        child = expand(child)
        print("Child Moves: ", child.moves)
        # Pick child from expanded set
        sim_idx = random.randint(0, len(child.moves)-1)
        simulate_child = child.moves[sim_idx]
        # Perform Simulation
        outcome = simulate(simulate_child.board)
    else:                       # Recurse if there are moves
        _, outcome, _ = mcts(child, state, False, 0) 
    if outcome == state.player:
        child.winner()
        if root:
            new_winner, win_ratio = update_selected(child, win_ratio)
    else:
        child.loser()
    parent.attempts += 1
    update_ucbs(parent)
    return new_winner, outcome, win_ratio


def find_best_move(state: GameState) -> None:
    """
    Search the game tree for the optimal move for the current player, storing
    the index of the move in the given GameState object's selected attribute.
    The move must be an integer indicating an index in the 3D board - ranging
    from 0 to 63 - with 0 as the index of the top-left space of the top board
    and 63 as the index of the bottom-right space of the bottom board.

    This function must perform a Monte Carlo Tree Search to select a move,
    using additional functions as necessary. During the search, whenever a
    better move is found, the selected attribute should be immediately updated
    for retrieval by the instructor's game driver. Each call to this function
    will be given a set number of seconds to run; when the time limit is
    reached, the index stored in selected will be used for the player's turn.
    """
    win_ratio: float = 0

    # First create the root node for the game
    root = State(None, None, state)
    root = expand(root)
    root = run_each_move(root, state)
    for _, move in enumerate(root.moves):
        idx, win_ratio = update_selected(move, win_ratio)
        if idx != None:
            state.selected = idx
    print("After intial runs, selected: ", state.selected)
    
    while True:
    #for _ in range(500):
        idx, _, win_ratio = mcts(root, state, True, win_ratio)   
        if idx != None:
            state.selected = idx
            print("Updated Selected: ", state.selected)


def main() -> None:
    #board = ((0, 0, 0, 0,
    #          0, 0, None, None,
    #          0, None, 0, None,
    #          0, None, None, 0),) \
    #        + ((None,) * 16,) * 3
    board = ((None, None, 0, None,
              None, None, 0, None,
              0, 0, 0, 0,
              None, None, 0, None),) \
            + ((None,) * 16,) * 3
    """
    board = ((0, 0, 0, 0,
              0, None, None, None,
              0, None, None, None,
              0, None, None, None),
             (None, None, None, None,
              None, None, None, None,
              None, None, None, None,
              None, None, None, None),
             (None, None, None, None,
              None, None, None, None,
              None, None, None, None,
              None, None, None, None),
             (0, 0, 0, 0,
              0, 0, None, None,
              0, None, 0, None,
              0, None, None, 0),)
    board = ((0, 0, 0, 0,
              0, None, None, None,
              0, None, None, None,
              0, None, None, None),
             (None, None, None, None,
              None, None, None, None,
              None, None, None, None,
              None, None, None, None),
             (None, None, None, None,
              None, None, None, None,
              None, None, None, None,
              None, None, None, None),
             (0, 0, 0, 0,
              0, 0, None, None,
              0, None, 0, None,
              0, None, None, 0),)
    """
    state = GameState(board, 1)
    print(state.display)
    find_best_move(state)
    print("Final Answer: ", state.selected)
    assert state.selected == 10
    #play_game()



def play_game() -> None:
    """
    Play a game of 3D Tic-Tac-Toe with the computer.

    If you lose, you lost to a machine.
    If you win, your implementation was bad.
    You lose either way.
    """
    board = tuple(tuple(0 for _ in range(i, i + 16))
                  for i in range(0, 64, 16))
    state = GameState(board, 1)
    while state.util is None:
        # human move
        print(state.display)
        state = state.traverse(int(input("Move: ")))
        if state.util is not None:
            break
        # computer move
        find_best_move(state)
        move = (state.selected if state.selected != -1
                else random.choice(state.moves))
        state = state.traverse(move)
    print(state.display)
    if state.util == 0:
        print("Tie Game")
    else:
        print(f"Player {state.util} Wins!")


if __name__ == "__main__":
    main()
