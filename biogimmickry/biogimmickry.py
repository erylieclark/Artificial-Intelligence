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
        del weights[1]
        del chars[1]
    elif last_char == ">":
        del weights[0]
        del chars[0]
    elif last_char == "+":
        del weights[3]
        del chars[3]
    elif last_char == "-":
        del weights[2]
        del chars[2]
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

    # Make the selection of characters to choose from
    if loop:        # A Loop should be involved
        all_chars: List[str] = ["<", ">", "+", "-", "[", "]"]
        all_weights: List[int] = [1, 1, 8, 8, 4, 4]
        char_choice = all_chars[1:4]
        weight_choice = all_weights[1:4]
    else:           # No loop involved
        all_chars: List[str] = ["<", ">", "+", "-"]
        all_weights: List[int] = [1, 1, 8, 8]
        char_choice = all_chars[1:]
        weight_choice = all_weights[1:]
    # Starting with a "<" or "[" is useless and cannot start with "]"

    for i in range(1,length):
        # Fill in the next program character
        program = program + random.choices(char_choice, \
            weights = weight_choice)[0]
        
        char_choice = all_chars.copy()
        weight_choice = all_weights.copy()
        char_choice, weight_choice = check_last_move(char_choice, \
            weight_choice, program[-1], loop)

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


def sort_and_slice(parents: List[str], scores: List[float], pcnt_keep: int) \
-> Tuple[List[str], List[float]]:
    """
    Sort the parents and then use the threshold and the scores to get rid of
    the lowest probabilities that the program will be used for creating the
    next generation.
    """
    #print("Num parents started with: ", len(scores))
    #for i in range(len(scores)):
    #    print("Parent: ", parents[i])
    #    print("    Score: {:0.2f}".format(scores[i]))
    #print("************************* Sorting **********************")
    # Sort the parents based on the scores
    parents = [x for _, x in sorted(zip(scores, parents), reverse = True)]
    # Sort the scores
    scores.sort(reverse = True)
    #for i in range(len(scores)):
        #print("Parent: ", parents[i])
        #print("    Score: {:0.3f}".format(scores[i]))

    #split_idx = list(filter(lambda i: i < th, scores))[0]
    #split_idx = next(x for x, val in enumerate(scores) if val < th)
    split_idx = round(len(scores)*pcnt_keep)
    #print("Split_idx: ", split_idx)
    #print("************************ Slicing ***********************")
    parents = parents[:split_idx]
    scores = scores[:split_idx]
    #for i in range(len(scores)):
    #    print("Parent: ", parents[i])
    #    print("    Score: {:0.2f}".format(scores[i]))
    #print(" Num parents ending with: ", len(scores))
    
    return parents, scores


def crossover(fe: FitnessEvaluator, parents: List[str], scores: List[float], \
elite_pcnt: int) -> List[str]:
    """
    Keep the top "elite" percent and transfer them directly to the next
    generation, then take the remainder and use them for crossover.
    """
    children: List[str] = [""]*len(parents)
    #child_scores: List[int] = [0]*len(parents)
    p1: str = ""
    p2: str = ""
    c1: str = ""
    c2: str = ""
    #print("Parents: ", parents)
    num_elite = round(len(parents)*elite_pcnt)
    # If remainder is odd, add one to elite so that crossover has even
    if (len(parents) - num_elite) % 2:
        num_elite = num_elite + 1
    # Copy the elites to the children
    children[:num_elite] = parents[:num_elite]
    #child_scores[:num_elite] = scores[:num_elite]
    #print("Children: ", children)
    # Use the parents to create new children using crossover
    for i in range(int((len(parents)-num_elite)/2)):
        p1, p2 = random.choices(parents, k=2, weights=scores)
        # Cross the pairs
        split_idx = random.randint(0, min(len(p1), len(p2)))
        #print("Split idx: ", split_idx)
        c1 = p1[:split_idx] + p2[split_idx:]
        c2 = p2[:split_idx] + p1[split_idx:]
        children[i*2+num_elite] = c1
        children[i*2+1+num_elite] = c2
        
    return children


def mutate_weird_sequence(program: str, loop: int) -> str:
    """
    Mutate -> Increase the mutation rate as the population gets smaller
    """
    sus_seqs: List[str] = ["<<", "<>", "><", ">>", "[]"]
    chars: List[str] = ["+", "-"]
    idx: int = -1
    
    if (idx := program.find("<>")) != -1:
        program = program[:idx] + program[idx-1] + \
            program[idx+1:]
    if (idx := program.find("><")) != -1:
        program = program[:idx] + program[idx-1] + \
            program[idx+1:]
    if (idx := program.find("<<")) != -1:
        program = program[:idx] + program[idx-1] + \
            program[idx+1:]
    if (idx := program.find(">>")) != -1:
        program = program[:idx] + program[idx-1] + \
            program[idx+1:]
    if loop:
        if (idx := program.find("[]")) != -1:
            program = program[:idx+1] + random.choice(chars)[0] + \
                program[idx+1:]
        
    return program
        
def mutate(children: List[str], init_pop: int, cur_pop: int, loop: int) \
-> List[str]:
    """
    Mutate -> Increase the mutation rate as the population gets smaller
    """
    sus_seqs: List[str] = ["<<", "<>", "><", ">>", "[]"]
    mutate: str = ""
    num_mutate = round((init_pop - cur_pop)/init_pop*cur_pop)
    print("Current Pop: ", cur_pop)
    print("Num Mutations: ", num_mutate)
    for i in range(num_mutate):
        mutate = random.choice(children)
        print("String to mutate: ", mutate)
        if any(word in mutate for word in sus_seqs):
            print("Found Sus Sequence.")
            mutate = mutate_weird_sequence(mutate, loop)
            print("Mutated Sus Sequence: ", mutate)



    return children


def create_program(fe: FitnessEvaluator, max_len: int) -> str:
    """
    Return a program string no longer than max_len that, when interpreted,
    populates a memory array that exactly matches a target array.

    Use fe.evaluate(program) to get a program's fitness score (zero is best).
    """
    init_pop: int = 500
    parents: List[str] = [""]*init_pop
    scores: List[int] = [0]*init_pop
    norm_scores: List[float] = []
    total: int = 0
    pcnt_keep: float = 0.98
    elitism: float = 0.02
    j: int = 0
    loop: int = False
    count: int = 0

    if max_len:
        loop = True
    else:
        max_len = 30
        loop = False
        
    # Get the parents for the starting population    
    for i in range(len(parents)):
        parents[i] = create_parent(max_len, loop)   # Create the parent
    
    while True:
        #print("-------------------------- New Loop ---------------------")
        #print("Length of population: ", len(parents))
        j = 0
        count = 0
        while j < len(parents):
            try:
                scores[j] = fe.evaluate(parents[j])     # Evaluate it
            except RunTimeError:
                print("RunTime Error")
                del parents[j]
                scores.pop()
                count += 1
                continue
            if not scores[j]:   # If we get lucky with one that succeeds
                return parents[j]
            j += 1
        #print(scores)
        #print("Count: ", count)
        #print("Parent Length: ", len(parents))
        #print("Scores Length: ", len(scores))
        # Add the scores up and normalize them
        norm_scores = normalize_scores(scores)

        # Sort the parents based on their scores
        # Use a threshold to weed some out
        parents, norm_scores = sort_and_slice(parents, norm_scores, pcnt_keep)

        # Pick pairs out of the rest to use for crossover
        children = crossover(fe, parents, norm_scores, elitism)
        children = mutate(children, init_pop, len(children), loop)
        parents = children
        # Shorten the scores to the length of the children
        scores = scores[:len(children)]
        #print("Children: ", children)
        # Mutate some of the crossed pairs
        # Evaluate new set
    
    
    return parents[0]

def main() -> None:  # optional driver
    array = (-1, 2, -3, 4)
    max_len = 0  # no BF loop required

    # only attempt when non-loop programs work
#    array = (20, 0)
#    max_len = 15

    program = create_program(FitnessEvaluator(array), max_len)
    print("Final Program: ", program)
    if max_len > 0:
        assert len(program) <= max_len
    assert array == FitnessEvaluator.interpret(program, len(array))


if __name__ == "__main__":
    main()
