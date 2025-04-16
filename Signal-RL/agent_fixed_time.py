# Agent algorithms:
# "Grad"/"SemiGrad": gradient or semi-gradient learning
# "DQN"/"SARSA": Q-learning or SARSA learning
# "Diff": differential learning
# "PPO": Proximal Policy Optimization
# "SAC": Soft Actor-Critic
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Load config file
with open("config.yaml", "r") as file:  
    config = yaml.safe_load(file)
env_step_duration = config['env_step_duration']

class FixedTimingSignalAgent:
    """
    Fixed Timing Signal Agent for benchmarking.
    Cycles through predefined traffic light phases in a fixed order.
    """

    def __init__(self):
        # Define the phase durations and the sequence
        # Each phase corresponds to a duration and a state
        # phase 0: N-S Through
        # phase 1: N-S Yellow
        # phase 2: N-S Left
        # phase 4: E-W Through
        # phase 6: E-W Left
        self.duration = np.asarray([34, 15, 34, 15])  # default: [45, 20, 45, 20];, [38, 17, 38, 17], L : T = 1 : 2.25
        self.duration = self.duration + 3  # add yellow
        self.end = np.cumsum(self.duration)
        self.begin = self.end - self.duration
        self.cycle = sum(self.duration)
        self.step = 0
        '''
        self.phases = [
            {"duration": 35 + 3, "state": "GGGrrrrrGGGrrrrr"},
            {"duration": 15 + 3, "state": "rrrGrrrrrrrGrrrr"},
            {"duration": 35 + 3, "state": "rrrrGGGrrrrrGGGr"},
            {"duration": 15 + 3, "state": "rrrrrrrGrrrrrrrG"},
        ]
        '''
        # time -> action
        # 0 -> 0
        # 38 -> 2
        # 56 -> 4
        # 94 -> 6
        # Note: "-3" since you need to propose early to activate yellow
        
    def take_action(self, state):
        """
        Returns the action (phase index) for the fixed timing agent.
        Cycles through the phases in a fixed order based on the phase duration.

        Returns:
            int: Current phase index.
        """
        step_mod = self.step % self.cycle
        current_phase_index = np.where(step_mod >= self.begin)[0][-1]
        self.step += env_step_duration
        return current_phase_index

    def update(self, transition_dict):
        pass

    def save(self, path):
        pass
    
    def load(self, path):
        pass