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


class Q_Func:

    def __init__(self, state: ModuleState) -> None:
        self.alt_bins = self._create_bins(state.altitude)
        self.vel_bins: List[List[int]] = [[100, 0], [0, -1], [-1, -100]]
        self.table = self._init_table(self.alt_bins, self.vel_bins, state.actions)


    def _init_table(self, alt_bins: List[List[float]], vel_bins: List[List[int]],\
        actions: List[int]) -> Dict:
        """
        Initialize the lookup table that will carry the utility values
        """
        action_dict: Dict = {}
        vel_dict: Dict = {}
        table: Dict = {}

        for i in range(len(actions)):
            action_dict[actions[i]] = 0.0

        for i in range(len(vel_bins)):
            vel_dict[tuple(vel_bins[i])] = action_dict

        for i in range(len(alt_bins)):
            table[tuple(alt_bins[i])] = vel_dict
        return table
    

    def _create_bins(self, alt: float) -> List[List[float]]:
        """
        Define the bins that will be used in this simulation
        """
        print("----------------- Creating Bins --------------------")
        # Could specify the number of bins and percent bins more to the
        #   initial state if this is too generic/hard-coded
        num_bins: int = 5           # Number of bins to represent the altitude
        pcnt_bins: List[float] = [1, 0.5, 0.25, 0.15, 0.05, 0]
        alt_bins: List[List[float]] = [[0,0]]*num_bins
        highest_alt: int = num_bins*(round(int(alt)/num_bins))
        print("Highest Altitude", highest_alt)
        for i in range(num_bins):
            alt_bins[i] = [highest_alt*pcnt_bins[i], highest_alt*pcnt_bins[i+1]]
        print("Bins", alt_bins)
        return alt_bins
    

    def get(self, arg: Tuple[ModuleState, int]) -> float:
        """
        Return the utility of a certain action
        """
        state = arg[0]
        action = arg[1]
        for key in self.table:
            if state.altitude <= key[0] and state.altitude >= key[1]:
                vel_dict = self.table[key]
        for key in vel_dict:
            if state.velocity <= key[0] and state.velocity >= key[1]:
                act_dict = vel_dict[key]
        if not action in act_dict:
            return 0.0
        #print("Act_dict[action]: ", act_dict[action])
        return act_dict[action]
    

    def calc_utility(self, state: ModuleState, action: int, lr: float, \
        df: float, R: float) -> float:
        """
        Return the utility of a certain action
        """
        for key in self.table:
            if state.altitude <= key[0] and state.altitude >= key[1]:
                vel_dict = self.table[key]
        for key in vel_dict:
            if state.velocity <= key[0] and state.velocity >= key[1]:
                act_dict = vel_dict[key]
        temp1 = (1 - lr)*act_dict[action]
        #temp2 = lr*(R + df*max(
        

def learn_q(state: ModuleState) -> Callable[[ModuleState, int], float]:
    """
    Return a Q-function that maps a state-action pair to a utility value. This
    function must be a callable with the signature (ModuleState, int) -> float.

    Optional: Use |state.set_actions| to set the size of the action set. Higher
    values offer more control (sensitivity to differences in rate changes), but
    require larger Q-tables and thus more training time.
    """
    print("----------------------------------------------")
    print("Actions: ", state.actions)
    print("New State:\n\r", state.use_fuel(1))
    print("----------------------------------------------")
    iterations: int = 100
    lr: float = 0.5                 # Learning Rate
    df: float = 0.95                # Discounting Factor
    starting_state = state          # Copy to return to on each iteration
    table = Q_Func(state)
    for _ in range(iterations):
        state = starting_state
        break
    return lambda s, a: table.get((s, a))
        


def main() -> None:
    fuel: int = 100
    altitude: float = 50.0

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
