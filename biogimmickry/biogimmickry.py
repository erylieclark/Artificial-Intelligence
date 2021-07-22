# Name:         Erin Rylie Clark
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Biogimmickry
# Term:         Summer 2021

import random
from typing import Callable, Dict, Tuple, List


class FitnessEvaluator:

    def __init__(self, array: Tuple[int, ...]) -> None:
        """
        An instance of FitnessEvaluator has one attribute, which is a callable.

            evaluate: A callable that takes a program string as its only
                      argument and returns an integer indicating how closely
                      the program populated the target array, with a return
                      value of zero meaning the program was accurate.

        This constructor should only be called once per search.

        >>> fe = FitnessEvaluator((0, 20))
        >>> fe.evaulate(">+")
        19
        >>> fe.evaulate("+++++[>++++<-]")
        0
        """
        self.evaluate: Callable[[str], int] = \
            lambda program: FitnessEvaluator._evaluate(array, program)

    @staticmethod
    def interpret(program: str, size: int) -> Tuple[int, ...]:
        """
        Using a zeroed-out memory array of the given size, run the given
        program to update the integers in the array. If the program is
        ill-formatted or requires too many iterations to interpret, raise a
        RuntimeError.
        """
        p_ptr = 0
        a_ptr = 0
        count = 0
        max_steps = 1000
        i_map = FitnessEvaluator._preprocess(program)
        memory = [0] * size
        while p_ptr < len(program):
            if program[p_ptr] == "[":
                if memory[a_ptr] == 0:
                    p_ptr = i_map[p_ptr]
            elif program[p_ptr] == "]":
                if memory[a_ptr] != 0:
                    p_ptr = i_map[p_ptr]
            elif program[p_ptr] == "<":
                if a_ptr > 0:
                    a_ptr -= 1
            elif program[p_ptr] == ">":
                if a_ptr < len(memory) - 1:
                    a_ptr += 1
            elif program[p_ptr] == "+":
                memory[a_ptr] += 1
            elif program[p_ptr] == "-":
                memory[a_ptr] -= 1
            else:
                raise RuntimeError
            p_ptr += 1
            count += 1
            if count > max_steps:
                raise RuntimeError
        return tuple(memory)

    @staticmethod
    def _preprocess(program: str) -> Dict[int, int]:
        """
        Return a dictionary mapping the index of each [ command with its
        corresponding ] command. If the program is ill-formatted, raise a
        RuntimeError.
        """
        i_map = {}
        stack = []
        for p_ptr in range(len(program)):
            if program[p_ptr] == "[":
                stack.append(p_ptr)
            elif program[p_ptr] == "]":
                if len(stack) == 0:
                    raise RuntimeError
                i = stack.pop()
                i_map[i] = p_ptr
                i_map[p_ptr] = i
        if len(stack) != 0:
            raise RuntimeError
        return i_map

    @staticmethod
    def _evaluate(expect: Tuple[int, ...], program: str) -> int:
        """
        Return the sum of absolute differences between each index in the given
        tuple and the memory array created by interpreting the given program.
        """
        actual = FitnessEvaluator.interpret(program, len(expect))
        return sum(abs(x - y) for x, y in zip(expect, actual))

def check_last_move(chars: List[str], weights: List[int], last_char: str, 
loop: int) -> Tuple[List[str], List[int]]:
    """
    Make sure that the next step does not undo the previous step.
    """
    if loop:
        if last_char == "[": # Loop started, replace [ with ] in choices
            del weights[4]
            del chars[4]
        if last_char == "]": # Loop ended, remove ] from choices
            del weights[5]
            del chars[5]
        if last_char == "<":
            del weights[2]
            del chars[2]
        if last_char == ">":
            del weights[3]
            del chars[3]
    elif last_char == "+":
        del weights[1]
        del chars[1]
    elif last_char == "-":
        del weights[0]
        del chars[0]
    return chars, weights


def create_parent(max_len: int, loop: int) -> str:
    """
    Max_len in this function is to determine the longest length a program
    should be started as. If a max_len is given to create_program, it will
    be passed directly, otherwise it will be set to an appropriate length.
    The function will create an arbitrary program of length max_len.
    """
    length: int = random.randint(1, max_len)
    program: str = ""
    all_chars: List[str] = []
    all_weights: List[int] = []
    loop_ended: int = False
    i: int = 0
    #char_choice: List[str] = []
    #weight_choice: List[int] = []

    # Make the selection of characters to choose from
    if loop:        # A Loop should be involved
        all_chars = ["+", "-", ">", "<", "[", "]"]
        all_weights = [8, 8, 1, 1, 4, 4]
        char_choice = all_chars[0:3]
        weight_choice = all_weights[0:3]
    else:           # No loop involved
        all_chars = ["+", "-", ">"]
        all_weights = [3, 3, 1]
        char_choice = all_chars
        weight_choice = all_weights
        # loop_ended = True
    # Starting with a "<" or "[" is useless and cannot start with "]"

    while i < length or loop_ended:
        # Fill in the next program character
        program = program + random.choices(char_choice, \
            weights=weight_choice)[0]
        
        char_choice = all_chars.copy()
        weight_choice = all_weights.copy()
        char_choice, weight_choice = check_last_move(char_choice, \
            weight_choice, program[-1], loop)
        if program[-1] == "]":
            loop_ended = True
        i += 1

    return program


def normalize_scores(scores: List[int]) -> List[float]:
    """
    Add the scores up and normalize them between 1 and 0
    """
    total: int = 0
    norm: List[float] = [0.0]*len(scores)

    # Add up all the scores
    for i in range(len(scores)):
        total = total + scores[i]
    # Subtract so that high scores are better
    for i in range(len(scores)):
        scores[i] = total - scores[i]
    # Divide all by the total to get probability
    for i in range(len(scores)):
        norm[i] = scores[i]/total
    
    return norm 


def sort_and_slice(parents: List[str], scores: List[float], pcnt_keep: float) \
-> Tuple[List[str], List[float]]:
    """
    Sort the parents and then use the threshold and the scores to get rid of
    the lowest probabilities that the program will be used for creating the
    next generation.
    """
    # Sort the parents based on the scores
    parents = [x for _, x in sorted(zip(scores, parents), reverse=True)]
    # Sort the scores
    scores.sort(reverse=True)
    split_idx = round(len(scores)*pcnt_keep)
    parents = parents[:split_idx]
    scores = scores[:split_idx]
    
    return parents, scores


def cross_no_loop(p1: str, p2: str) -> Tuple[str, str]:
    """
    Keep the top "elite" percent and transfer them directly to the next
    generation, then take the remainder and use them for crossover.
    """
    #i: int = 0 
    c1: str = ""
    c2: str = ""
    idx_list: List[int] = []
    #print("Parent 1: ", p1)
    #print("Parent 2: ", p2)

    #idx_list = [i for i, letter in enumerate(p1) if letter == ">"]
    if len(idx_list) < 1:
        idx = random.randint(0, len(p1))
    else:
        idx = random.choice(idx_list)
    #print("P1 split idx: ", idx)
    c1 = p1[:idx] 
    c2 = p1[idx:]
    #idx_list = [i for i, letter in enumerate(p2) if letter == ">"]
    if len(idx_list) < 1:
        idx = random.randint(0, len(p2))
    else:
        idx = random.choice(idx_list)
    #print("P2 split idx: ", idx)
    c1 = c1 + p2[idx:] 
    c2 = p2[:idx] + c2
    #print("C1 after P2: ", c1)
    #print("C2 after P2: ", c2)
    return c1, c2


def crossover(fe: FitnessEvaluator, parents: List[str], scores: List[float], \
elite_pcnt: float, loop: int, max_len: int) -> List[str]:
    """
    Keep the top "elite" percent and transfer them directly to the next
    generation, then take the remainder and use them for crossover.
    """
    children = parents
    p1: str = ""
    p2: str = ""
    c1: str = ""
    c2: str = ""
    #print("Parents: ", parents)
    num_elite = round(len(parents)*elite_pcnt)
    # If remainder is odd, add one to elite so that crossover has even
    if (len(parents) - num_elite) % 2:
        num_elite = num_elite + 1
    # Use the parents to create new children using crossover
    for i in range(int((len(parents)-num_elite)/2)):
        p1, p2 = random.choices(parents, k=2, weights=scores)
        # Cross the pairs
        if loop:
            print("Contains a loop.")
            #c1, c2 = cross_loop(p1, p2)
        else:
            c1, c2 = cross_no_loop(p1, p2)
            if len(c1) > max_len:
                c1 = c1[:max_len]
            if len(c2) > max_len:
                c2 = c2[:max_len]
        children[i*2+num_elite] = c1
        children[i*2+1+num_elite] = c2
        
    return children


def mutate_weird_sequence(program: str, loop: int) -> str:
    """
    Mutate -> Increase the mutation rate as the population gets smaller
    """
    idx = program.find("<>")
    if idx != -1:
        program = program[:idx] + program[idx-1] + \
            program[idx+1:]
    idx = program.find("><")
    if idx != -1:
        program = program[:idx] + program[idx-1] + \
            program[idx+1:]
    idx = program.find("+-+")
    if idx != -1:
        program = program[:idx+1] + program[idx] + \
            program[idx+2:]
    idx = program.find("-+-")
    if idx != -1:
        program = program[:idx+1] + program[idx] + \
            program[idx+2:]
    idx = program.find("+-")
    if idx != -1:
        program = program[:idx]
    idx = program.find("-+")
    if idx != -1:
        program = program[:idx]
    #if (idx := program.find("<<")) != -1:
    #    program = program[:idx] + program[idx-1] + \
    #        program[idx+1:]
    #if (idx := program.find(">>")) != -1:
    #    program = program[:idx] + program[idx-1] + \
    #        program[idx+1:]
    if loop:
        idx = program.find("[]")
        if idx != -1:
            program = program[:idx+1] + random.choice(["+", "-"])[0] + \
                program[idx+1:]
        
    return program

        
def mutate_add(program: str, loop: int) -> str:
    """
    Mutate
    """
    all_chars: List[str] = [""]
    all_weights: List[int] = [0]
    i: int = 0
    idx: int = 0
    num_muts: int = round(len(program)*0.1)
    
    if loop:
        all_chars = ["+", "-", ">", "<", "[", "]"]
        all_weights = [8, 8, 1, 1, 4, 4]
    else:           # No loop involved
        all_chars = ["+", "-", ">"]
        all_weights = [1, 1, 1]
    
    while i < num_muts:
        idx = random.randint(0, len(program))
        add_char = random.choices(all_chars, weights=all_weights)[0]
        program = program[:idx] + add_char + program[idx:]
        i += 1
    
    return program



def mutate_remove(program: str, loop: int) -> str:
    """
    Mutate 
    """
    num_muts: int = round(len(program)*0.1)
    i: int = 0
    idx: int = 0
    while i < num_muts:
        idx = random.randint(0, len(program))
        program = program[:idx] + program[idx+1:]
        i += 1
    return program


def mutate_replace(program: str, loop: int) -> str:
    """
    Mutate 
    """
    all_chars: List[str] = [""]
    all_weights: List[int] = [0]
    i: int = 0
    idx: int = 0
    num_muts: int = round(len(program)*0.1)
    
    if loop:
        all_chars = ["+", "-", ">", "<", "[", "]"]
        all_weights = [8, 8, 1, 1, 4, 4]
    else:           # No loop involved
        all_chars = ["+", "-", ">"]
        all_weights = [3, 3, 1]
    
    while i < num_muts:
        idx = random.randint(0, len(program))
        add_char = random.choices(all_chars, weights=all_weights)[0]
        program = program[:idx] + add_char + program[idx+1:]
        i += 1

    return program



def mutate(children: List[str], init_pop: int, cur_pop: int, loop: int) \
-> List[str]:
    """
    Mutate -> Increase the mutation rate as the population gets smaller
    """
    sus_seqs: List[str] = ["<>", "><", "+-+", "-+-", "+-", "-+", "[]"]
    muts: List[str] = ["+", "-", "r"]
    weights: List[int] = []
    mutate: str = ""
    idx: int = 0
    num_mutate = round((init_pop - cur_pop)/init_pop*cur_pop)

    if loop:
        weights = [3, 6, 2]
    else: 
        weights = [6, 1, 4]
    #print("Current Pop: ", cur_pop)
    #print("Num Mutations: ", num_mutate)
    for _ in range(num_mutate):
        idx = random.choice(range(len(children)))
        mutate = children[idx]
        #print("String to mutate: ", mutate)
        if any(word in mutate for word in sus_seqs):
            #print("Found Sus Sequence.")
            mutate = mutate_weird_sequence(mutate, loop)
            #print("Mutated Sus Sequence: ", mutate)
        next_mut = random.choices(muts, weights=weights)[0]
        #print("Next mutation: ", next_mut)
        if next_mut == "-":
            mutate = mutate_remove(mutate, loop)
        elif next_mut == "+":
            mutate = mutate_add(mutate, loop)
        elif next_mut == "r":
            mutate = mutate_replace(mutate, loop)
        children[idx] = mutate

    return children


def eval_new_parents(fe: FitnessEvaluator, parents: List[str], \
scores: List[int]) -> Tuple[List[str], List[int], int]:
    """
    evaluate the new parents created
    """
    j: int = 0

    while j < len(parents):
        try:
            scores[j] = fe.evaluate(parents[j])     # Evaluate it
        except RuntimeError:
            del parents[j]
            scores.pop()
            continue
        if not scores[j]:   # If we get lucky with one that succeeds
            return parents, scores, j
        j += 1
    return parents, scores, -1


def check_startover(parents: List[str], scores: List[int], count: int, \
init_pop: int, max_len: int, loop: int) -> Tuple[List[str], List[int], int]:
    """
    check_startover - checks for convergence of the scores to be within too
    small of a range, or if the population size has gotten too small. Will
    start over if either of these occur
    """
    init_parents: List[str] = [""]*init_pop
    init_scores: List[int] = [0]*init_pop

    # Check for convergence by seeing if the range of scores is small
    if max(scores)-min(scores) < 5:
        count += 1      # Allow it to continue a few more times
    else:
        count = 0
    if len(parents) < round(init_pop*0.01) or count > 10: 
        parents = init_parents.copy()   # Reset the parents to init pop size
        scores = init_scores.copy()     # Reset scores to init pop size
        count = 0                       # Reset means not converging anymore
        for i in range(init_pop):       # Create the new parents
            parents[i] = create_parent(max_len, loop)  
            
    #print("Initial Population: ", parents)
    return parents, scores, count


def create_program(fe: FitnessEvaluator, max_len: int) -> str:
    """
    Return a program string no longer than max_len that, when interpreted,
    populates a memory array that exactly matches a target array.

    Use fe.evaluate(program) to get a program's fitness score (zero is best).
    """
    init_pop: int = 1500
    parents: List[str] = [""]
    scores: List[int] = [0]
    #avg_score: float = 100
    count: int = 0
    #pcnt_keep: float = 0.98
    #elitism: float = 0.02
    idx: int = 0
    loop: int = False

    if max_len:
        loop = True
        return "+"
    else:
        max_len = 40
        
    # Get the parents for the starting population    
    
    while True:
        if len(parents) < round(init_pop*0.01) or (max(scores)-min(scores)) < 5:
            parents, scores, count = check_startover(parents, scores, count,\
                init_pop, max_len, loop)
        
        # Evaluate the newly generated parents and return the scores
        parents, scores, idx = eval_new_parents(fe, parents, scores)
        if idx != -1:
            return parents[idx]
        #print(scores)

        # Add the scores up and normalize them
        norm_scores = normalize_scores(scores)

        # Sort the parents based on their scores
        # Use a threshold to weed some out
        #print("Parent Length before slice: ", len(parents))
        parents, norm_scores = sort_and_slice(parents, norm_scores, 0.98)
        #print("Parent Length after slice: ", len(parents))

        # Pick pairs out of the rest to use for crossover
        children = crossover(fe, parents, norm_scores, 0.05, loop, max_len)
        
        # Mutate some of the crossed pairs
        parents = mutate(children, init_pop, len(children), loop)
        
        # Shorten the scores to the length of the children
        scores = scores[:len(children)]
        #print("Children: ", children)
    
    
    return parents[0]

def main() -> None:  # optional driver
    array = (-1, 2, -3, 4, -4, 3, 2) #, -4, 3, 2, -1)
    max_len = 0  # no BF loop required

    # only attempt when non-loop programs work
#    array = (20, 0)
#    max_len = 15

    program = create_program(FitnessEvaluator(array), max_len)
    if max_len > 0:
        assert len(program) <= max_len
    assert array == FitnessEvaluator.interpret(program, len(array))
    print("Final Program: ", program)


if __name__ == "__main__":
    main()
