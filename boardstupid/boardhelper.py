# Name:         Erin Rylie Clark
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Board Stupid
# Term:         Summer 2021

import math
import random
import boardstupid
from typing import Callable, Generator, Optional, Tuple, List


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

def expand(child):
    """
    Get the moves that the child can make and select one at random
    """
    C: float = 2 ** 0.5 # Start with sqrt(2)

    print("======================================Expanding===========================================")
    state = child.board
    print("Child Moves: ", state.moves)
    for _, m in enumerate(state.moves):
        board = state.traverse(m)
        # TODO: Check if the board is solved
        print("New Move: ", m)
        print("board: ", board.display)
        new_move = State(m, C, board)   # Add index of the move and C as the UCB
        child.add_move(new_move, C)      # Add the object to the list of moves
    return child


def simulate(state):
    """
    Get the moves that the child can make and select one at random
    """
    print("=======================================Simulating=========================================")
    moves = list(state.moves)
    print("    moves available: ", moves)
    for i in range(len(state.moves)):
        move = random.choice(moves)
        print("        move making: ", move)
        move_idx = moves.index(move)
        print("        index of move: ", move_idx)
        moves.pop(move_idx)
        print("        new moves available: ", moves)
        state = state.traverse(move)
        print("         New Board: ", state.display)
    print("    Winner: ", state.util)
    return state.util
   


def MCTS(parent):
    """
    Perform Monte Carlo Tree Search
    """
    # Find the maximum UCB value to explore
    idx = parent.move_ucbs.index(max(parent.move_ucbs))
        # TODO: May want to make this a random selection instead of first
    # Select the child with the highest UCB value
    child = parent.moves[idx]
    print("Child Selected: ", child.idx)
    # Check if the child has moves expanded already
    if child.attempts == 0 or len(child.moves) == 0: # Expand if no moves
        child = expand(child)
        # Pick child from expanded set
        child_idx = random.randint(0, len(child.moves)-1)
        simulate_child = child.moves[child_idx]
        # Perform Simulation
        outcome = simulate(simulate_child.board)

    else:                       # Recurse if there are moves
        child = MCTS(child) 
    # 



def find_best_move(state) -> None:
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
    C: float = 2 ** 0.5 # Start with sqrt(2)
    idx: int = -1

    # First create the root node for the game
    root = State(None, None, state)
    print("Moves: ", state.moves)
    for _, m in enumerate(state.moves):
        board = state.traverse(m)
        print("New Move: ", m)
        print("board: ", board.display)
        new_move = State(m, C, board)   # Add index of the move and C as the UCB
        root.add_move(new_move, C)      # Add the object to the list of moves

    while True:
        MCTS(root)   
        
        # Pick one of the highest ranked by UCB's
        break
        pass






