# Name:         Erin Rylie Clark
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Moonlander II
# Term:         Summer 2021

import random
from typing import Callable, Tuple, Dict, List


class ModuleState:  # do not modify class

    def __init__(self, fuel: int, altitude: float, force: float,
                 transition: Callable[[float, float], float],
                 velocity: float = 0.0,
                 actions: Tuple[int, ...] = tuple(range(5))) -> None:
        """
        An instance of ModuleState has the following attributes.

            fuel: The amount of fuel (in liters) able to be used.
            altitude: The distance (in meters) of the module from the surface
                      of its target object.
            velocity: The speed of the module, where a positive value indicates
                      movement away from the target object and a negative value
                      indicates movement toward it. Defaults to zero.
            actions: The available fuel rates, where 0 indicates free-fall and
                     the highest-valued action indicates maximum thrust away
                     from the target object. Defaults to (0, 1, 2, 3, 4).
            use_fuel: A callable that takes an integer as its only argument to
                      be used as the fuel rate for moving to the next state.
        """
        self.fuel: int = fuel
        self.altitude: float = altitude
        self.velocity: float = velocity
        self.actions: Tuple[int, ...] = actions
        self.use_fuel: Callable[[int], ModuleState] = \
            lambda rate: self._use_fuel(force, transition, rate)

    def __repr__(self) -> str:
        if not self.altitude:
            return ("-" * 16 + "\n"
                    + f" Remaining Fuel: {self.fuel:4} l\n"
                    + f"Impact Velocity: {self.velocity:7.2f} m/s\n")
        else:
            return (f"    Fuel: {self.fuel:4} l\n"
                    + f"Altitude: {self.altitude:7.2f} m\n"
                    + f"Velocity: {self.velocity:7.2f} m/s\n")

    def set_actions(self, n: int) -> None:
        """
        Set the number of actions available to the module simulator, which must
        be at least two. Calling this method overrides the default number of
        actions set in the constructor.

        >>> module.set_actions(8)
        >>> module.actions
        (0, 1, 2, 3, 4, 5, 6, 7)
        """
        if n < 2:
            raise ValueError
        self.actions = tuple(range(n))

    def _use_fuel(self, force: float, transition: Callable[[float, int], float],
                  rate: int) -> "ModuleState":
        """
        Return a ModuleState instance in which the fuel, altitude, and velocity
        are updated based on the fuel rate chosen.

        Do not call this method directly; instead, call the |use_fuel| instance
        attribute, which only requires a fuel rate as its argument.
        """
        if not self.altitude:
            return self
        fuel = max(0, self.fuel - rate)
        if not fuel:
            rate = 0
        acceleration = transition(force * 9.8, rate / (len(self.actions) - 1))
        altitude = max(0.0, self.altitude + self.velocity + acceleration / 2)
        velocity = self.velocity + acceleration
        return ModuleState(fuel, altitude, force, transition, velocity=velocity,
                           actions=self.actions)


class QFunc:

    def __init__(self, state: ModuleState) -> None:
        self.alt_bins = self._create_alt_bins(state.altitude)
        #self.vel_bins: List[List[int]] = [[100, 0], [0, -1], [-1, -100]]
        self.vel_bins: List[List[int]] = self._create_vel_bins(state.altitude)
        self.table = self._init_table(self.alt_bins, self.vel_bins, \
            state.actions)


    def _init_table(self, alt_bins: List[List[float]], \
        vel_bins: List[List[int]], actions: List[int]) -> Dict:
        """
        Initialize the lookup table that will carry the utility values
        """
        action_dict: Dict = {}
        vel_dict: Dict = {}
        table: Dict = {}

        action_dict[actions[0]] = 0.5
        for i in range(1, len(actions)):
            action_dict[actions[i]] = 0.1

        for i in range(len(vel_bins)):
            vel_dict[tuple(vel_bins[i])] = action_dict.copy()

        for i in range(len(alt_bins)):
            table[tuple(alt_bins[i])] = vel_dict.copy()
            for j in range(len(vel_bins)):
                table[tuple(alt_bins[i])][tuple(vel_bins[j])] = \
                    action_dict.copy()
            
        return table
    

    def _create_vel_bins(self, alt: float) -> List[List[float]]:
        """
        Define the bins that will be used in this simulation
        """
        vel_bins = [[100, 0], [0, -0.5], [-0.5, -1], [-1, -2], [-2, -5], \
            [-5, -100]]
        return vel_bins
    

    def _create_alt_bins(self, alt: float) -> List[List[float]]:
        """
        Define the bins that will be used in this simulation
        """
        # Number of bins depending on the altitude to represent altitude
        num_bins: int = round((alt + 2.5)/5) 
        # The bins will not be evenly spaced, smaller with less altitude
        r: float = 0.4                      # Rate of exponential decay
        pcnt_bins: List[float] = [round(1.5*r**i, 5) for i in range(num_bins)]
        pcnt_bins.append(0)                 # Add 0 to finish off the list
        # Initialize the bins
        alt_bins: List[List[float]] = [[0, 0]]*num_bins
        # Determine start and stop points for each bin
        for i in range(num_bins):
            alt_bins[i] = [round(alt*pcnt_bins[i], 5), \
                round(alt*pcnt_bins[i+1], 5)]
        return alt_bins
    

    def get(self, arg: Tuple[ModuleState, int]) -> float:
        """
        Return the utility of a certain action
        """
        state = arg[0]
        action = arg[1]
        act_dict = self.get_action_set(state)
        if not action in act_dict:
            return 0.0
        return act_dict[action]

    
    def get_action_set(self, state: ModuleState) -> Dict:
        """
        Return the utility of a certain action
        """
        act_dict: Dict = {}
        for key in self.table:
            if state.altitude <= key[0] and state.altitude >= key[1]:
                vel_dict = self.table[key]
                for key in vel_dict:
                    if state.velocity <= key[0] and state.velocity >= key[1]:
                        act_dict = vel_dict[key]
                        break
                break
        return act_dict
    

    def calc_utility(self, state: ModuleState, action: int, lr: float, \
        df: float, r: float) -> float:
        """
        Return the utility of a certain action
        """
        act_dict: Dict = {}
        alt: Tuple[int, ...] = (0, 0)
        vel: Tuple[int, ...] = (0, 0)
        for alt in self.table:
            if state.altitude <= alt[0] and state.altitude >= alt[1]:
                vel_dict = self.table[alt]
                for vel in vel_dict:
                    if state.velocity <= vel[0] and state.velocity >= vel[1]:
                        act_dict = vel_dict[vel]
                        break
                break
        action_set = self.get_action_set(state.use_fuel(action))
        if len(action_set) != len(state.actions) or len(act_dict) < 1:
            return
        else:
            temp1 = (1 - lr)*(act_dict.get(action))
            temp2 = lr*(r + df*max(list(action_set.values())))
            total = temp1 + temp2
            self.table[alt][vel][action] = total
    

    def print_table(self) -> None:
        """
        print
        """
        for alt in self.table:
            print("Altitude Range: ", alt)
            vel_dict = self.table[alt]
            for vel in vel_dict:
                print("    Velocity Range: ", vel)
                act_dict = vel_dict[vel]
                print("        Action Utilities: ", list(act_dict.values()))


def reward_function(state: ModuleState, max_alt: float) -> Tuple[float, int]:
    """
    Reward Function
    """
    pos: float = 1.0
    neg: float = -1.0
    non: float = -0.04
    term = False
    if not state.altitude:      # Terminal State
        if state.velocity > -1:
            r = pos
        else:
            r = neg
        term = True
    elif state.altitude > max_alt:
        r = neg
    else:
        if state.velocity > 0:
            r = -0.2
        else:
            r = non
    return r, term


def run_simulation(state: ModuleState, table: QFunc, starting_alt: float, \
lr: float, df: float, epsilon: float) -> None:
    """
    Run Simulation
    """
    weights = [0.7, 0.3, 0.2, 0.1, 0.1]
    act_choice: int = 0
    while True:
        # Choose the action to take
        act_set = table.get_action_set(state)
        act_choice = random.choices((0, 1), weights=[epsilon, 1-epsilon])[0] 
        if act_choice == 1 or len(list(act_set.values())) < 1:
            action = random.choices(state.actions, weights=weights)[0]
        else:
            action = max(act_set, key=act_set.get)
        # Get the reward, terminate if term state???
        r, term = reward_function(state, starting_alt)
        # Calculate the utility
        if state.altitude < starting_alt*1.3:
            # Make sure we are not going up/outside of bounds
            table.calc_utility(state, action, lr, df, r)
            # Take the action
        state = state.use_fuel(action)
        if term:
            break


def update_df(alt: float, start_alt: float) -> float:
    """
    Update Discounting Factor
    """
    starting_df: float = 0.1
    max_df: float = 0.95

    df = starting_df + round(alt/start_alt, 3)

    if df > max_df:
        df = max_df
    
    return df


def update_lr(cur_iter: int, total_iters: int) -> float:
    """
    Update Learning Rate
    """
    start_lr = 0.9
    lowest_lr = 0.6
    lr = start_lr - (start_lr - lowest_lr)*(cur_iter/total_iters)
    return lr


def update_epsilon(cur_iter: int, total_iters: int) -> float:
    """
    Update Learning Rate
    """
    start = 0.2
    highest = 0.9
    epsilon = start + (cur_iter/total_iters)
    if epsilon > highest:
        epsilon = highest
    return epsilon 


def learn_q(state: ModuleState) -> Callable[[ModuleState, int], float]:
    """
    Return a Q-function that maps a state-action pair to a utility value. This
    function must be a callable with the signature (ModuleState, int) -> float.

    Optional: Use |state.set_actions| to set the size of the action set. Higher
    values offer more control (sensitivity to differences in rate changes), but
    require larger Q-tables and thus more training time.
    """
    iterations: int = 5000
    #start_iters: int = 100
    lr: float = 0.8                 # Learning Rate
    df: float = 0.95                # Discounting Factor
    epsilon: float = 0.2
    
    starting_state = state          # Copy to return to on each iteration
    #start_alt = starting_state.altitude
    table = QFunc(state)

    # Start by filling out the state space near the planet    
    # Keep learning rate high for this part
    '''
    for alt in range(int(start_alt)):
        df = update_df(alt, start_alt)
        for i in range(start_iters):
            #epsilon = update_epsilon(i, start_iters)
            state = starting_state
            state.altitude = alt
            run_simulation(state, table, start_alt, lr, df, epsilon)
    '''

    #print("Starting main iterations")
    for i in range(iterations):
        state = starting_state
        lr = update_lr(i, iterations)
        epsilon = update_epsilon(i, iterations)
        run_simulation(state, table, starting_state.altitude, lr, df, epsilon)

    return lambda s, a: table.get((s, a))
        


def main() -> None:
    fuel: int = 200
    altitude: float = 75.0

    gforces = {"Pluto": 0.063, "Moon": 0.1657, "Mars": 0.378, "Venus": 0.905,
               "Earth": 1.0, "Jupiter": 2.528}
    transition = lambda g, r: g * (2 * r - 1)  # example transition function

    state = ModuleState(fuel, altitude, gforces["Moon"], transition)
    q = learn_q(state)
    policy = lambda s: max(state.actions, key=lambda a: q(s, a))

    print(state)
    while state.altitude > 0:
        state = state.use_fuel(policy(state))
        print(state)


if __name__ == "__main__":
    main()
