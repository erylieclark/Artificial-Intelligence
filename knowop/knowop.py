# Name:         Erin Rylie Clark
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Know Op
# Term:         Summer 2021

import math
import itertools
import random
from typing import Callable, Dict, List, Tuple


class Math:
    """A collection of static methods for mathematical operations."""

    @staticmethod
    def dot(xs: List[float], ys: List[float]) -> float:
        """Return the dot product of the given vectors."""
        return sum(x * y for x, y in zip(xs, ys))

    @staticmethod
    def matmul(xs: List[List[float]],
               ys: List[List[float]]) -> List[List[float]]:
        """Multiply the given matrices and return the resulting matrix."""
        product = []
        for x_row in range(len(xs)):
            row = []
            for y_col in range(len(ys[0])):
                col = [ys[y_row][y_col] for y_row in range(len(ys))]
                row.append(Math.dot(xs[x_row], col))
            product.append(row)
        return product

    @staticmethod
    def transpose(matrix: List[List[float]]) -> List[List[float]]:
        """Return the transposition of the given matrix."""
        return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

    @staticmethod
    def relu(z: float) -> float:
        """
        The activation function for hidden layers.
        """
        return z if z > 0 else 0.01 * z

    @staticmethod
    def relu_prime(z: float) -> float:
        """
        Return the derivative of the ReLU function.
        """
        return 1.0 if z > 0 else 0.0

    @staticmethod
    def sigmoid(z: float) -> float:
        """
        The activation function for the output layer.
        """
        epsilon = 1e-5
        return min(max(1 / (1 + math.e ** -z), epsilon), 1 - epsilon)

    @staticmethod
    def sigmoid_prime(z: float) -> float:
        """
        The activation function for the output layer.
        """
        return Math.sigmoid(z) * (1 - Math.sigmoid(z))

    @staticmethod
    def loss(actual: float, expect: float) -> float:
        """
        Return the loss between the actual and expected values.
        """
        return -(expect * math.log10(actual)
                 + (1 - expect) * math.log10(1 - actual))

    @staticmethod
    def loss_prime(actual: float, expect: float) -> float:
        """
        Return the derivative of the loss.
        """
        return -expect / actual + (1 - expect) / (1 - actual)




class Layer:  # do not modify class

    def __init__(self, size: Tuple[int, int], is_output: bool) -> None:
        """
        Create a network layer with size[0] levels and size[1] inputs at each
        level. If is_output is True, use the sigmoid activation function;
        otherwise, use the ReLU activation function.

        An instance of Layer has the following attributes.

            g: The activation function - sigmoid for the output layer and ReLU
               for the hidden layer(s).
            w: The weight matrix (randomly-initialized), where each inner list
               represents the incoming weights for one neuron in the layer.
            b: The bias vector (zero-initialized), where each value represents
               the bias for one neuron in the layer.
            z: The result of (wx + b) for each neuron in the layer.
            a: The activation g(z) for each neuron in the layer.
           dw: The derivative of the weights with respect to the loss.
           db: The derivative of the bias with respect to the loss.
        """
        self.g = Math.sigmoid if is_output else Math.relu
        self.w: List[List[float]] = \
            [[random.random() * 0.1 for _ in range(size[1])]
             for _ in range(size[0])]
        self.b: List[float] = [0.0] * size[0]

        # use of below attributes is optional but recommended
        self.z: List[float] = [0.0] * size[0]
        self.a: List[float] = [0.0] * size[0]
        self.dw: List[List[float]] = \
            [[0.0 for _ in range(size[1])] for _ in range(size[0])]
        self.db: List[float] = [0.0] * size[0]

    def __repr__(self) -> str:
        """
        Return a string representation of a network layer, with each level of
        the layer on a separate line, formatted as "W | B".
        """
        s = "\n"
        fmt = "{:7.3f}"
        for i in range(len(self.w)):
            s += " ".join(fmt.format(w) for w in self.w[i])
            s += " | " + fmt.format(self.b[i]) + "\n"
        return s

    def activate(self, inputs: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Given an input (x) of the same length as the number of columns in this
        layer's weight matrix, return g(wx + b).
        """
        self.z = [Math.dot(self.w[i], inputs) + self.b[i]
                   for i in range(len(self.w))]
        self.a = [self.g(real) for real in self.z]
        return tuple(self.a)




def create_samples(f: Callable[..., int], n_args: int, n_bits: int,
) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """
    Return a dictionary that maps inputs to expected outputs.
    """
    samples = {}
    max_arg = 2 ** n_bits
    for inputs in itertools.product((0, 1), repeat=n_args * n_bits):
        ints = [int("".join(str(bit) for bit in inputs[i:i + n_bits]), 2)
                for i in range(0, len(inputs), n_bits)]
        try:
            output = f(*ints)
            if 0 <= output < max_arg:
                bit_string = ("{:0" + str(n_bits) + "b}").format(output)
                samples[inputs] = tuple(int(bit) for bit in bit_string)
        except ZeroDivisionError:
            pass
    return samples


def init_layers(num_layers: int, num_levels: int, i_size: int, o_size: int) \
-> List[Layer]:
    """
    Initialize each layer with a non zero weight matrix and a zero bias
    """
    network: List[Layer] = []
    print("Initializing Layers")
    if num_layers > 1:
        # First layer based on input size
        network.append(Layer((num_levels, i_size), False))

        # Middle Layers - each have same levels in and out
        for i in range(1, num_layers - 1):
            network.append(Layer((num_levels, num_levels), True))

        # Last layer - has num levels going in, output size coming out
        network.append(Layer((o_size, num_levels), True))
    else:
        network.append(Layer((o_size, i_size), True))
        
    return network


def get_cost(actual: Tuple[float], expect: Tuple[float]) \
-> float:
    """
    Get loss of each individual output neruon
    """
    loss: float = 0.0
    cost: float = 0.0
    for i in range(len(actual)):
        loss = loss + Math.loss(actual[i], expect[i])
    cost = loss/len(actual)
    return cost


def add_weights(list1: List[List[float]], list2: List[List[float]]) \
-> List[List[float]]:
    """
    Update the weight matrices
    """
    new_list: List[List[float]] = []
    assert len(list1) == len(list2)
    assert len(list1[0]) == len(list2[0])
    # Multiply list 2 by -1
    for i in range(len(list1)):
        new_list.append([sum(x) for x in zip(list1[i], list2[i])])
    return new_list


def update_lr(cur_batch: int, args: int) \
-> float:
    """
    update the learning rate
    """
    start_lr: float = 0.5 
    rate: float = 0.99 
    pcnt: float = rate**(cur_batch)
    lr: float = pcnt*start_lr
    return lr


def get_loss_prime(actual: Tuple[float], expect: Tuple[float]) \
-> Tuple[float]:
    """
    Get loss of each individual output neruon
    """
    loss: List[float] = []
    for i in range(len(actual)):
        loss.append(Math.loss_prime(actual[i], expect[i]))
    return loss 


def get_active_func(func: Callable, vals: List[float]) \
-> Tuple[float]:
    """
    Return a vector containing the results from the given function
    """
    new_vals: List[float] = []
    for i in range(len(vals)):
        new_vals.append(func(vals[i]))
    return new_vals


def backprop(network: List[Layer], layers: int, y: List[float], \
sample: List[int], expect: List[float]) -> Tuple[float]:
    """
    Back prop
    """
    da = get_loss_prime(y, expect) 
    gpz = get_active_func(Math.sigmoid_prime, network[layers-1].z)
    for i in range(layers-1, -1, -1):
        temp_dz = [a*b for a,b in zip(da, gpz)]
        dz = [[el] for el in temp_dz]
        if i > 0:
            network[i].dw = add_weights(Math.matmul(dz, [network[i-1].a]), \
                network[i].dw)
        else:
            network[i].dw = add_weights(Math.matmul(dz, [sample]), \
                network[i].dw)
        network[i].db = [sum(x) for x in zip(network[i].db, \
            list(itertools.chain(*dz)))]
        if i >= 0:
            da = Math.matmul(Math.transpose(network[i].w), dz)
            # Flatten the list
            da = list(itertools.chain(*da))
            gpz = get_active_func(Math.relu_prime, network[i-1].z)
        else:
            break


def update_network(network: List[Layer], lr: float, num_layers: int) \
-> List[Layer]:
    """
    Update weights and biases
    """
    for layer in range(num_layers):
        dw = network[layer].dw
        change_w = [[ -1 * lr * w for w in inner ] for inner in dw]
        network[layer].w = add_weights(network[layer].w, change_w)
        db = network[layer].db
        change_b = [ -1 * lr * b for b in db ]
        # Set to a 0 list before the start of next batch
        network[layer].b = \
            [sum(x) for x in zip(network[layer].b, change_b)]
        network[layer].db = [0 * b for b in network[layer].db ]
        network[layer].dw = \
            [[0 * w for w in inner ] for inner in network[layer].dw]
    return network

def train_network(samples: Dict[Tuple[int, ...], Tuple[int, ...]],
                  i_size: int, o_size: int) -> List[Layer]:
    """
    Given a training set (with labels) and the sizes of the input and output
    layers, create and train a network by iteratively propagating inputs
    (forward) and their losses (backward) to update its weights and biases.
    Return the resulting trained network.
    """
    num_args: int = int(i_size/o_size)
    num_layers: int = 1 
    num_levels: int = 8 
    num_batches: int = 100 if num_args == 1 else 300
    max_cost: float = 0.09
    batch_size: int = 100 
    lr: float = 0.0
    cost: float = 0.0
    #print("Batch Size: ", batch_size)

    # Initialize the network
    network = init_layers(num_layers, num_levels, i_size, o_size)
    # Forward propagate for batch_size samples
    for b in range(num_batches):
        lr = update_lr(b, num_args)
        for i in range(batch_size):
            input_vec = random.choice(list(samples.keys()))
            expect_out = samples[input_vec]
            # y is the final output of forward propagation
            y = propagate_forward(network, input_vec)
            cost = cost + get_cost(y, expect_out)
            # Now backprop it
            backprop(network, num_layers, y, input_vec, expect_out)
        cost = cost/batch_size
        if(cost < max_cost):
            #print("Cost is sufficiently low. Stop Training.")
            return network
        #print("Avg Cost: ", cost)
        cost = 0.0
        # Average out db and dw
        for layer in range(num_layers):
            network[layer].db = [db/batch_size for db in network[layer].db]
            network[layer].dw = [[ dw/batch_size for dw in inner ] for \
                inner in network[layer].dw]
        # Update the weights and biases
        network = update_network(network, lr, num_layers)
            
    return network

    
def propagate_forward(layers: List[Layer], sample: List[int]) \
-> Tuple[float]:
    """
    Propagate Forward
    """
    layer_input = sample
    for i in range(len(layers)):
        layer_input = layers[i].activate(layer_input)
    
    return layer_input


def count_matches(inputs: Tuple[int], outputs: Tuple[int]) \
-> int:
    """
    Propagate Forward
    """
    cnt = 0
    for i in range(len(inputs)):
        if inputs[i] == outputs[i]:
            cnt += 1
    return cnt


def main() -> None:
    random.seed(0)
    #f = lambda x: 130  # operation to learn
    #f = lambda x, y: x + y   # operation to learn
    f = lambda x: x // 2  # operation to learn
    #f = lambda x, y: x  # operation to learn
    #f = lambda x, y: x | y  # operation to learn
    #f = lambda x, y: x & y  # operation to learn
    n_args = 1              # arity of operation
    n_bits = 8              # size of each operand
    pcnt_right = 0
    total_right = 0
    total = 0

    samples = create_samples(f, n_args, n_bits)
    train_pct = 0.99
    train_set = {inputs: samples[inputs]
               for inputs in random.sample(list(samples),
                                           k=int(len(samples) * train_pct))}
    test_set = {inputs: samples[inputs]
               for inputs in samples if inputs not in train_set}
    print("Train Size:", len(train_set), "Test Size:", len(test_set))

    network = train_network(train_set, n_args * n_bits, n_bits)
    print("*************************** TESTING *******************************")
    for inputs in test_set:
        output = tuple(round(n, 2) for n in propagate_forward(network, inputs))
        bits = tuple(round(n) for n in output)
        print("OUTPUT:", output)
        print("BITACT:", bits)
        print("BITEXP:", samples[inputs], end="\n")
        total_right = total_right + count_matches(bits, samples[inputs]) 
        total += n_bits
        print("Match: ", count_matches(bits, samples[inputs]), end="\n\n")
    print("Pcnt Right: ", total_right/total*100)
    print(network)

if __name__ == "__main__":
    main()
