# Name:         Erin Rylie Clark
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Board Stupid
# Term:         Summer 2021

import math
import random
from boardstupid import *
from typing import Callable, Generator, Optional, Tuple, List



def expand(child: State) -> State:
    """
    Get the moves that the child can make and select one at random
    """
    C: float = 2 ** 0.5 # Start with sqrt(2)

    state = child.board
    #print("Child Moves: ", state.moves)
    for _, m in enumerate(state.moves):
        board = state.traverse(m)
        # TODO: Check if the board is solved
        #print("New Move: ", m)
        #print("board: ", board.display)
        new_move = State(m, C, board)   # Add index of the move and C as the UCB
        child.add_move(new_move, C)      # Add the object to the list of moves
    return child


def simulate(state: GameState) -> int:
    """
    Get the moves that the child can make and select one at random
    """
    moves = list(state.moves)
    #print("    moves available: ", moves)
    for i in range(len(state.moves)):
        move = random.choice(moves)
        #print("        move making: ", move)
        move_idx = moves.index(move)
        #print("        index of move: ", move_idx)
        moves.pop(move_idx)
        #print("        new moves available: ", moves)
        state = state.traverse(move)
    #print("    Winner: ", state.util)
    #print("         New Board: ", state.display)
    return state.util
   

def update_ucbs(parent: State, C: float = 2 ** 0.5):
    for i, child in enumerate(parent.moves): 
        if child.attempts > 0 :
            #print("    Old UCB: ", child.ucb)
            #print("    Parent Attempts: ", parent.attempts)
            #print("    Self Attempts: ", child.attempts)
            #print("    Wins: ", child.wins)
            #print("    C: ", C)
            fraction = (math.log(parent.attempts)/child.attempts) ** 0.5
            child.ucb = child.wins/child.attempts + C * fraction
            #print("    New UCB: ", child.ucb)
            parent.move_ucbs[i] = child.ucb

def update_selected(state: GameState, move: State, win_ratio: float) -> float:
    ratio: float = move.wins/move.attempts
    if ratio > win_ratio:
        state.selected = move.idx
        win_ratio = ratio
        print(state.selected)
        
    return win_ratio


def run_each_move(root: State, state: GameState) -> State:
    """
    Run each initial move a number of times to get a better idea
    of which ones will work best
    """
    for m, base_move in enumerate(root.moves):
        for _ in range(50):
            base_move = expand(base_move)
            outcome, _ = MCTS(base_move, state, False, 0)
            if outcome == state.player:
                base_move.winner()
            else:
                base_move.loser()
            root.attempts += 1
        root.moves[m] = base_move
        print("Move: ", base_move.idx)
        print("    Wins: ", base_move.wins)
    update_ucbs(root)
    return root 
    

def MCTS(parent: State, state: GameState, root: int, win_ratio: float) -> int:
    """
    Perform Monte Carlo Tree Search
    """
    #print("\nPrior Move UCBs: ")
    #for m in range(len(parent.move_ucbs)):
    #    print("    ", parent.move_ucbs[m])
    # Find the maximum UCB value to explore
    idx = parent.move_ucbs.index(max(parent.move_ucbs))
        # TODO: May want to make this a random selection instead of first
    # Select the child with the highest UCB value
    child = parent.moves[idx]
    #print("Child Selected: ", child.idx)
    # Check if the child has moves expanded already
    if child.attempts == 0 or len(child.moves) == 0: # Expand if no moves
        child = expand(child)
        # Pick child from expanded set
        sim_idx = random.randint(0, len(child.moves)-1)
        simulate_child = child.moves[sim_idx]
        # Perform Simulation
        outcome = simulate(simulate_child.board)
    else:                       # Recurse if there are moves
        #print("Recursing.................................")
        outcome, _ = MCTS(child, state, False, 0) 
    # Update the wins/attempts
    #print("Outcome: ", outcome)
    #print("Player: ", state.player)
    if outcome == state.player:
        #print("********** Winner **********")
        #print("Index in list for ucb: ", idx)
        child.winner()
        if root:
            win_ratio = update_selected(state, child, win_ratio)
    else:
        #print("********** Loser **********")
        #print("Index in list for ucb: ", idx)
        child.loser()
    parent.attempts += 1
    update_ucbs(parent)
    #print("New Move UCBs: ")
    #for m in range(len(parent.move_ucbs)):
    #    print("    ", parent.move_ucbs[m])
    

    return outcome, win_ratio



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
    win_ratio: float = 0

    # First create the root node for the game
    root = State(None, None, state)
    root = expand(root)
    root = run_each_move(root, state)
    print("New Move UCBs: ")
    for m in range(len(root.move_ucbs)):
        print("    ", root.move_ucbs[m])
    
    #while True:
    for _ in range(10000):
        _, win_ratio = MCTS(root, state, True, win_ratio)   
        #print("\n\n\nRoot Attempts: ", root.attempts)
        #print("Back in main\n\n\n")
        
        # Pick one of the highest ranked by UCB's






