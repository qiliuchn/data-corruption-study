# Isolated intersection environment, implementing Gym API
# throughput will be used as performance index
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import random
import yaml
from collections import defaultdict
import torch

# Load config file
with open("config.yaml", "r") as file:  
    config = yaml.safe_load(file)

#================configuration================
#SUMO_BINARY = "sumo"  # Use "sumo" for terminal mode; use "sumo-gui" for GUI mode; when running in parallel, use "sumo"!
env_name = config['env_name']
SUMO_CONFIG = env_name + ".sumocfg"  # sumo config file
if env_name == "junction_200":
    max_lane_length = 200
elif env_name == "junction_400":
    max_lane_length = 400
else:
    raise ValueError("Invalid env name")
run_mode = config['run_mode']
render_mode = 'human'#config['render_mode']  # 'human' or 'rgb_array'; when running in parallel, use "rgb_array"! rgb_array by default
env_step_duration = config['env_step_duration']
tl_id = "J3"  # Traffic light ID from your `tlLogic`
num_phases = 4  # num of phases (only count green phases)
green_phases = [0, 2, 4, 6]  # Green phases
yellow_phases = [1, 3, 5, 7]  # yellow phases; yellow phase index = green phase index + 1
length_per_cell = config['length_per_cell']
max_num_cells = max_lane_length // length_per_cell  # maximum number of “cells” (or segments) per lane; here 1 cell is 5 meter long; lane length = 200 m = 40 cellls
yellow = config['yellow']  # yellow time
min_green = config['min_green']  # minimum green time
max_green = config['max_green']  # maximum green time
num_bounds = 4  # num of bounds of the intersection
num_lanes_per_bound = 3  # number of lanes per bound
lane_ids = [f"E{i}_{j}" for i in range(num_bounds) for j in range(num_lanes_per_bound)]
lane_index_to_id = lane_ids
lane_id_to_index = {lane_id:i for i, lane_id in enumerate(lane_ids)}
# 'E0' for west
# 'E1' for east
# 'E2' for south
# 'E3' for north
# rightmost lanes are numbered '0';
num_lanes = 12  # total num of lanes
time_limit = config['time_limit']
speed_threshold = config['speed_threshold']
reward_type = config['reward_type']
corruption_mode = config['corruption_mode']
corruption_ratio = config['corruption_ratio']
imputation_mode = config['imputation_mode']
nearby_lanes = [[0,1]] * 2 + [[2]] \
    + [[3,4]] * 2 + [[5]] \
    + [[6,7]] * 2 + [[8]] \
    + [[9,10]] * 2 + [[11]] #for imputation
#==============================================

class TrafficSignalEnv(gym.Env):
    def __init__(self):
        # Note: put all traci calls to start_simulation()
        # so that sumo is started after sub-processes are created
        super().__init__()
        self.num_phases = num_phases
        self.num_lanes = num_lanes
        self.max_lane_length = max_lane_length
        self.length_per_cell = length_per_cell
        self.max_num_cells = max_num_cells
        self.min_green = min_green
        self.max_green = max_green
        self.yellow = yellow
        self.lane_ids = lane_ids
        self.lane_index_to_id = lane_index_to_id
        self.lane_id_to_index = lane_id_to_index
        self.tl_id = tl_id
        self.green_phases = green_phases
        self.yellow_phases = yellow_phases
        self.env_step_duration = env_step_duration
        self.speed_threshold = speed_threshold
        self.sumo_binary = "sumo-gui" if render_mode == 'human' else "sumo"
        self.run_mode = run_mode
        # action is the index of the green phase to display next step
        self.action_space = spaces.Discrete(self.num_phases)
        # state is flattened occupancy matrix + current phase index (one-hot) + time since last phase change (normalized)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.num_lanes * self.max_num_cells + self.num_phases + 1,), 
            dtype=np.float32
        )
        #self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_lanes, self.max_num_cells), dtype=np.int32)
        self.reward_type = reward_type
        self.sumo_config = SUMO_CONFIG
        self.time_limit = time_limit
        self.corruption_mode = corruption_mode
        self.corruption_ratio = corruption_ratio
        self.imputation_mode = imputation_mode
        self.nearby_lanes = nearby_lanes  # for imputation

    def seed(self, seed=None):
        # set seed for reproducibility
        # gym api
        random.seed(seed)
        np.random.seed(seed)
    
    def start_simulation(self):
        if traci.isLoaded():
            traci.close()
        traci.start([self.sumo_binary, "-c", self.sumo_config, "--no-step-log", "--no-warnings"], numRetries=200)  # add --no-warnings flag if desired
        self.current_phase = traci.trafficlight.getPhase(self.tl_id)
        self.time_since_last_phase_change = self.min_green
        self.in_transition = None  # whether the traffic light is in transition
        self.next_green_phase = None  # the next green phase while in transition
        self.step_count = 0
        self.cumulative_departed_vehicles = set()
        self.cumulative_active_vehicles = set()
        self.cumulative_loaded_vehicles = set()
        self.cumulative_teleported_vehicles = set()
        self.cumulative_arrived_vehicles = set()
        # Dictionary to store historical vehicle data
        self.vehicle_history = defaultdict(list)
        self.lane_lengths = [0] * len(self.lane_ids)
        for i, lane_id in enumerate(self.lane_ids):
            self.lane_lengths[i] = traci.lane.getLength(lane_id)
        self.last_step_vehicles = {}  # for calculating throughput reward
        self.vehicles_exited = set()  # for calculating throughput reward
        # for vehicle_miss corruption
        self.missing_vehicles = []

    def reset(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        if traci.isLoaded():
            traci.close()
        self.start_simulation()
        occupancy_matrix = np.zeros((len(self.lane_ids), self.max_num_cells), dtype=np.int32)
        current_phase_onehot = np.zeros(self.num_phases)
        current_phase_onehot[green_phases.index(self.current_phase)] = 1
        time_since_last_phase_change_normalized = (0 + self.yellow) / (self.max_green + self.yellow)
        state = np.concatenate((occupancy_matrix.flatten(), 
                        current_phase_onehot, 
                        np.asarray([time_since_last_phase_change_normalized])))
        #for testing
        #print(f"Reset called. State shape: {state.shape}")  # Debug statement
        #assert state.shape == (2405,), f"State shape mismatch: {state.shape}"
        return state, {}

    def log_vehicle_data(self):
        # Log vehicle data
        # for debugging
        # Vehicles specified in the demand file up to this step
        loaded_vehicles = traci.simulation.getLoadedIDList()
        self.cumulative_loaded_vehicles.update(loaded_vehicles)
        # Vehicles that entered the network in the current step
        departed_vehicles = traci.simulation.getDepartedIDList()
        self.cumulative_departed_vehicles.update(departed_vehicles)
        # Vehicles that are currently active in the network
        current_vehicles = traci.vehicle.getIDList()
        self.cumulative_active_vehicles.update(current_vehicles)
        # Vehicles that have been teleported in the current step
        teleported_vehicles = traci.simulation.getStartingTeleportIDList()
        self.cumulative_teleported_vehicles.update(teleported_vehicles)
        # Vehicles that have arrived in the current step
        arrived_vehicles = traci.simulation.getArrivedIDList()
        self.cumulative_arrived_vehicles.update(arrived_vehicles)
        # Log relevant information for the vehicle on the network
        for veh_id in current_vehicles:
            self.vehicle_history[veh_id].append({
                "step": self.step_count,
                "route": traci.vehicle.getRouteID(veh_id),
                "speed": traci.vehicle.getSpeed(veh_id),
                "position": traci.vehicle.getPosition(veh_id),
                "lane": traci.vehicle.getLaneID(veh_id),
                "waiting_time": traci.vehicle.getWaitingTime(veh_id),
                "edge": traci.vehicle.getRoadID(veh_id),
                "status": "Active"
            })
        # Log vehicles that have departed but not yet activated
        for veh_id in self.cumulative_departed_vehicles:
            if veh_id not in self.cumulative_active_vehicles:
                self.vehicle_history[veh_id].append({
                    "step": self.step_count,
                    "status": "Departed, but not yet active"})
        # Log teleported vehicles
        for veh_id in teleported_vehicles:
            self.vehicle_history[veh_id].append({
                "step": self.step_count,
                "status":"Teleported"})
        # Log vehicles that have arrived (finished their route)
        arrived_vehicles = traci.simulation.getArrivedIDList()
        for veh_id in arrived_vehicles:
            self.vehicle_history[veh_id].append({"step": self.step_count,
                                                    "status":"Arrived"})

    def get_road_occupancy_and_occupancy_reward(self):
        # get occupancy matrix
        occupancy_matrix = np.zeros((len(self.lane_ids), self.max_num_cells), dtype=np.int32)
        for i, lane_id in enumerate(self.lane_ids):
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicle_ids:
                position = traci.vehicle.getLanePosition(vehicle_id)
                cell_index = int(position / self.length_per_cell)
                cell_index = min(cell_index, self.max_num_cells - 1)
                occupancy_matrix[i, cell_index] = 1
        
        # get reward
        occupancy_value = np.sum(occupancy_matrix)  # Sum of occupied cells
        reward = occupancy_value  # Add waiting time to the reward
        # normalize reward to roughly in range [-1, 1]; important!
        reward = reward - 100
        reward = reward / 50
        reward = - reward
        return occupancy_matrix, reward

    def get_road_occupancy_and_throughput_reward(self):
        # get occupancy matrix
        current_vehicles = {}
        occupancy_matrix = np.zeros((len(self.lane_ids), self.max_num_cells), dtype=np.int32)
        for i, lane_id in enumerate(self.lane_ids):
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            current_vehicles[lane_id] = set(vehicle_ids)
            for vehicle_id in vehicle_ids:
                position = traci.vehicle.getLanePosition(vehicle_id)
                cell_index = int(position / self.length_per_cell)
                cell_index = min(cell_index, self.max_num_cells - 1)
                occupancy_matrix[i, cell_index] = 1
        
        # get reward
        throughput = 0
        for lane_id in self.lane_ids:
            last_step_vehicles = self.last_step_vehicles.get(lane_id, set())
            current_step_vehicles = current_vehicles.get(lane_id, set())
            # Vehicles that were in the lane last step but are no longer in the lane now
            exited_vehicles = last_step_vehicles - current_step_vehicles
            # Only count vehicles that have not been counted before
            # Simulations operate in discrete time steps. During each step, vehicle positions are updated, 
            # but vehicles may remain within the intersection area for multiple steps, leading to repeated detections.
            new_exited_vehicles = exited_vehicles - self.vehicles_exited
            throughput += len(new_exited_vehicles)
            # Update the global set of exited vehicles
            self.vehicles_exited.update(new_exited_vehicles)
        reward = throughput / 10  # Normalize throughput
        self.last_step_vehicles = current_vehicles  # Update for the next step
        return occupancy_matrix, reward
    

    def get_road_occupancy_and_waiting_time_reward(self):
        """
        Computes the road occupancy matrix and waiting time reward with optional corruption modes.
        """
        occupancy_matrix = np.zeros((len(self.lane_ids), self.max_num_cells), dtype=np.int32)
        waiting_vehicles = 0
        waiting_vehicles_test_mode = 0
        
        for i, lane_id in enumerate(self.lane_ids):
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)  # call for each lane
            for vehicle_id in vehicle_ids:
                #---------------corruption---------------
                if self.corruption_mode == "vehicle_miss" and vehicle_id in self.missing_vehicles:
                    # if vehicle is missing, it will not appear in the occupancy matrix nor in the reward
                    # except in test mode - occupancy is corrupted but reward is not
                    if self.run_mode == "test":
                        if traci.vehicle.getSpeed(vehicle_id) < self.speed_threshold:  # Check if vehicle is waiting
                            waiting_vehicles_test_mode += 1
                    continue
                position = traci.vehicle.getLanePosition(vehicle_id)  # call for each vehicle
                if self.corruption_mode == "region_mask":
                    # if a vehicle is within masked region, it will not appear in the occupancy matrix nor in the reward
                    if position < self.lane_lengths[i] * self.corruption_ratio:
                        if self.run_mode == "test":
                            if traci.vehicle.getSpeed(vehicle_id) < self.speed_threshold:  # Check if vehicle is waiting
                                waiting_vehicles_test_mode += 1
                        continue
                #----------------------------------------
                cell_index = int(position / self.length_per_cell)
                if cell_index < 0 or cell_index > self.max_num_cells:
                    raise ValueError(f"Invalid position location {position} for vehicle {vehicle_id} in lane {lane_id}")     
                cell_index = min(cell_index, self.max_num_cells - 1)
                occupancy_matrix[i, cell_index] = 1
    
                # check if vehicle is waiting
                # traci.vehicle.getSpeed returns the speed of the named vehicle within the last step [m/s]; error value: -2^30
                if traci.vehicle.getSpeed(vehicle_id) < self.speed_threshold:  # Check if vehicle is waiting
                    waiting_vehicles += 1
                    waiting_vehicles_test_mode += 1
                    
        # Normalize reward to roughly in range [-1, 1]
        reward = waiting_vehicles if self.run_mode == "train" else waiting_vehicles_test_mode
        reward = reward - 80  # Adjust for baseline waiting vehicles
        reward = reward / 80
        reward = -reward  # Flip the reward to make smaller queues better

        #---------------corruption---------------
        # Corrupt the occupancy matrix and reward if corruption_mode is insert_noise
        if self.corruption_mode == "insert_noise":
            # Set a fraction of cells in the occupancy matrix to noise
            noise_values = np.random.choice([0, 1], size=occupancy_matrix.shape)
            corruption_flag = np.random.random(size=occupancy_matrix.shape) < self.corruption_ratio
            occupancy_matrix[corruption_flag] = noise_values[corruption_flag]
            # corrupt the reward only if in train mode
            if self.run_mode == 'train':
                # corrupt the reward
                noise_percentage = np.random.uniform(-1, 1) * self.corruption_ratio + 1
                # reward cannot be larger than 1.0 and smaller than -3
                reward = np.clip(reward * noise_percentage, -3.0, 1.0)
        #-----------------------------------------
        
        #---------------imputation---------------
        # impute the occupancy mattrix and reward
        if self.imputation_mode == 'context_fill':
            prev_occupancy_matrix = self.context_mode_imputation(occupancy_matrix)
            while True:
                next_occupancy_matrix = self.context_mode_imputation(prev_occupancy_matrix)
                if np.array_equal(prev_occupancy_matrix, next_occupancy_matrix):
                    break
                else:
                    prev_occupancy_matrix = next_occupancy_matrix
            occupancy_matrix = prev_occupancy_matrix
        elif self.imputation_mode == 'moving_average':
            occupancy_matrix = self.moving_average_imputation(occupancy_matrix)
        elif self.imputation_mode == 'denosing_autoendoer':
            occupancy_matrix = self.denoising_autoencoder_imputation(occupancy_matrix)
        #-----------------------------------------
        return occupancy_matrix, reward


    def context_mode_imputation(self, occupancy_matrix):
        """
        Imputes missing values in the occupancy matrix using the median of the surrounding context.
        """
        occupancy_matrix_imputed = occupancy_matrix.copy()
        for i in range(len(self.lane_ids)):
            lane_level_context = np.zeros((1, occupancy_matrix.shape[1]))
            # Add context for the lane itself (downstream and upstream)
            downstream_cells = np.append(occupancy_matrix[i, 1:], 0)  # Shift right, add 0 at the end
            upstream_cells = np.insert(occupancy_matrix[i, :-1], 0, 0)  # Shift left, add 0 at the start
            lane_level_context += downstream_cells + upstream_cells
            
            for j in self.nearby_lanes[i]:
                if j != i:
                    lane_level_context += occupancy_matrix[j, :]
                    
            imputation_flag = (lane_level_context >= 2)
            occupancy_matrix_imputed[i, :] = np.where(imputation_flag, 1, occupancy_matrix[i, :])
        return occupancy_matrix_imputed

    def moving_average_imputation(self, occupancy_matrix):
        """
        Imputes values in the occupancy matrix using a moving average.
        """
        for i in range(len(self.lane_ids)):
            lane_values = occupancy_matrix[i, :]
            for j in range(self.max_num_cells):
                if lane_values[j] == 0:  # Missing value
                    # Compute moving average from neighbors
                    left = lane_values[max(0, j - 2):j]  # Two cells to the left
                    right = lane_values[j + 1:min(self.max_num_cells, j + 3)]  # Two cells to the right
                    neighbors = np.concatenate((left, right))
                    if len(neighbors) > 0:
                        occupancy_matrix[i, j] = np.mean(neighbors)  # Replace with mean of neighbors
        return occupancy_matrix

    def denoising_autoencoder_imputation(self, occupancy_matrix):
        """
        Imputes missing values in the occupancy matrix using a pre-trained denoising autoencoder.
        """
        if not hasattr(self, 'autoencoder_model'):
            raise ValueError("Denoising autoencoder model is not trained or loaded.")

        occupancy_flat = occupancy_matrix.flatten()
        occupancy_tensor = torch.tensor(occupancy_flat, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        imputed_tensor = self.autoencoder_model(occupancy_tensor)
        imputed_matrix = imputed_tensor.squeeze(0).detach().numpy().reshape(occupancy_matrix.shape)
        return np.round(imputed_matrix)  # Round to nearest integer (0 or 1)


    def step_per_second(self, action):
        '''
            env step for one second
            action: index of the green phase to display next step
            return: state, reward, done, info, {}
        '''
        self.apply_action(action)
        traci.simulationStep()
        #self.log_vehicle_data()
        
        # mask vehicles if corruption_mode is 'veh_miss'
        if self.corruption_mode == 'vehicle_miss':
            # find out the newly entered vehicles
            newly_entered_vehicles = traci.simulation.getDepartedIDList()  # Get the IDs of vehicles that entered the network
            # mask vehicles as missing randomly
            for vehicle_id in newly_entered_vehicles:
                if random.random() < self.corruption_ratio:
                    self.missing_vehicles.append(vehicle_id)
        
        # get occupancy matrix and reward
        if self.reward_type == 'waiting_time':
            occupancy_matrix, reward = self.get_road_occupancy_and_waiting_time_reward()
        elif self.reward_type == 'throughput':
            occupancy_matrix, reward = self.get_road_occupancy_and_throughput_reward()
        elif self.reward_type == 'occupancy':
            occupancy_matrix, reward = self.get_road_occupancy_and_occupancy_reward()
        else:
            raise ValueError('Invalid reward type')
        
        # create state
        # state is flattened occupancy matrix + current phase index (one-hot) + time since last phase change
        # state dim is (num_lanes * max_num_cells + num_phases + 1,)
        current_phase_onehot = np.zeros(self.num_phases)
        if self.in_transition is not None:
            current_phase_onehot[green_phases.index(self.next_green_phase)] = 1
        else:
            current_phase_onehot[green_phases.index(self.current_phase)] = 1
        time_since_last_phase_change_normalized = (self.time_since_last_phase_change + self.yellow) / (self.max_green + self.yellow)
        state = np.concatenate((occupancy_matrix.flatten(), 
                                current_phase_onehot, 
                                np.asarray([time_since_last_phase_change_normalized])))
        done = traci.simulation.getMinExpectedNumber() <= 0
        if self.step_count >= self.time_limit - 1:
            done = True
        if done:
            traci.close()
        self.step_count += 1
        return state, reward, done, False, {}

    def step(self, action):
        state_list = []  # save state list for checking
        reward_list = []
        done_flag = False
        # Repeat action for env_step_duration times
        for t in range(self.env_step_duration):
            state, reward, done, _, __ = self.step_per_second(action)
            state_list.append(state)
            reward_list.append(reward)
            if done:
                done_flag = True
                break
        # Normalize rewards to keep the reward in the duration to be around [-1, 1]
        if reward_type == 'waiting_time':
            reward_total = sum(reward_list) / self.env_step_duration
        elif reward_type == 'throughput':
            reward_total = sum(reward_list)
        else:  # occupancy reward
            reward_total = sum(reward_list) / self.env_step_duration  # normalize the reward
        # return the last state, total reward, and done flag
        # Note: if the process terminate in the middle of the env_step_duration, the reward will be incomplete!
        return state_list[-1], reward_total, done_flag, False, {}

    def apply_action(self, action):
        if self.in_transition is not None:  # if in transition
            self.in_transition -= 1
            if self.in_transition > 0:  # if still in transition
                traci.trafficlight.setPhase(self.tl_id, self.current_phase)  # keep current phase
                self.time_since_last_phase_change += 1
                return
            else:  # transition finished
                traci.trafficlight.setPhase(self.tl_id, self.next_green_phase)  # go to next phase
                self.current_phase = self.next_green_phase
                self.in_transition = None
                self.next_green_phase = None
                self.time_since_last_phase_change += 1
                return

        if self.time_since_last_phase_change < self.min_green:  # if not in transition, and not long enough
            self.time_since_last_phase_change += 1
            traci.trafficlight.setPhase(self.tl_id, self.current_phase)
            return

        # if not in transition and green long enough
        phase_index = self.green_phases[action]
        if phase_index == self.current_phase:  # if next phase is the same with current phase
            if self.time_since_last_phase_change < self.max_green:  # if max green not reached
                traci.trafficlight.setPhase(self.tl_id, self.current_phase)  # continue current phase
                self.time_since_last_phase_change += 1
                return
            else:  # if current phase has been running for max_green
                if phase_index in [0, 4]:
                    phase_index = (phase_index + 4) % (2 * self.num_phases)  # if current phase is through, go to next through green (opposite road)
                else:
                    phase_index = (phase_index + 2) % (2 * self.num_phases)  # if current phase is left turn, go to next through green (opposite road)
                self.next_green_phase = phase_index  # remember next phase, then start yellow
                traci.trafficlight.setPhase(self.tl_id, self.current_phase + 1)  # Note: here we assume that yellow phasse has index 1 + current phase
                self.in_transition = self.yellow  # in transition for next yellow seconds
                self.time_since_last_phase_change = - self.yellow  # time since last phase start from (- yellow) second
                self.current_phase = self.current_phase + 1
                return
        else:  # if next phase is different with current phase (, and current phase is longer than min_green)
            self.next_green_phase = phase_index
            traci.trafficlight.setPhase(self.tl_id, self.current_phase + 1)  # Note: here we assume that yellow phasse has index 1 + current phase
            self.in_transition = self.yellow
            self.time_since_last_phase_change = - self.yellow
            self.current_phase = self.current_phase + 1
            return
    
    def render(self, mode='rgb_array'):
        self.sumo_binary = "sumo-gui" if render_mode == 'human' else "sumo"

    def close(self):
        if traci.isLoaded():
            traci.close()



#================Main Script========================
if __name__ == '__main__':
    # for test only
    import time
    
    # Initialize the environment
    env = TrafficSignalEnv()
    #env.sumo_binary = "sumo-gui"
    env.corruption_mode = 'vehicle_miss'
    env.imputation_mode = 'context_fill'

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    # Reset the environment
    state, _ = env.reset()
    print("\nInitial state shape:", state.shape)
    print("Initial state (first 10 elements):", state[:10])

    total_reward = 0

    for step in range(10000):
        action = env.action_space.sample()  # Sample a random action
        state, reward, done, truncated, info = env.step(action)  # Perform action

        print(f"\nStep {step + 1}")
        print("Action:", action)
        print("Reward:", reward)
        print("Done:", done)
        print("Truncated:", truncated)
        print("State shape:", state.shape)

        total_reward += reward
        if done:
            print("Simulation ended.")
            break

    print("\nTotal reward:", total_reward)

    # Close the environment
    env.close()