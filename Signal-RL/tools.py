# Utils
from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from environment import TrafficSignalEnv

# Callback function for SB3
def make_env(seed=None):
    """
    Creates a single environment instance for vectorized environments.
    """
    def _init():
        env = TrafficSignalEnv()
        if seed is not None:
            env.seed(seed)
        return env
    return _init

# Replay buffer for off-policy algorithms
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.stack(state), np.array(action), np.array(reward), np.stack(next_state), np.array(done)

    def size(self):
        return len(self.buffer)
    
    def __getitem__(self, key):
        if isinstance(key, slice):  # Handle slicing
            sliced_buffer = list(self.buffer)[key.start:key.stop:key.step]
        elif isinstance(key, int):  # Handle single indexing
            sliced_buffer = [self.buffer[key]]
        else:
            raise TypeError("Invalid index type. Must be int or slice.")
        
        # Extract components
        state, action, reward, next_state, done = zip(*sliced_buffer)
        return (
            np.stack(state),
            np.array(action),
            np.array(reward),
            np.stack(next_state),
            np.array(done),
        )

# Epsilon-greedy exploration scheduler
def get_epsilon_linear(episode, epsilon_initial, epsilon_final, total_episodes=500):
    """
    Linear epsilon decay function.
    """
    decay_rate = (epsilon_initial - epsilon_final) / total_episodes
    epsilon = max(epsilon_final, epsilon_initial - decay_rate * episode)
    return epsilon

def get_epsilon_exponential(episode, epsilon_initial, epsilon_final, total_episodes=500):
    decay_rate = (epsilon_final / epsilon_initial) ** (1 / total_episodes)
    epsilon = max(epsilon_final, epsilon_initial * (decay_rate ** episode))
    return epsilon

def get_epsilon_dynamic(episode, epsilon_initial, epsilon_final, total_episodes=500):
    decay_steps = int(total_episodes * 0.8)
    # Dynamic Decay with Early Plateau: For traffic signal optimization tasks, ensure that exploration doesn’t decay too quickly
    if episode < decay_steps:
        return epsilon_initial - (episode / decay_steps) * (epsilon_initial - epsilon_final)
    else:
        return epsilon_final

# LR schedulers
class LRSchedulerConstant:
    def __init__(self, optimizer, lr_initial, lr_final=None, total_episodes=500):
        """
        Custom Exponential Learning Rate Scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be updated.
            initial_lr (float): Initial learning rate at the beginning of training.
            final_lr (float): Final learning rate at the end of training.
            total_steps (int): Total number of steps (or episodes) during training.
        """
        self.total_episodes = total_episodes
        self.optimizer = optimizer
        self.lr_initial = lr_initial
        self.lr_final = lr_final
        self.current_episode = 0
        self.current_lr = lr_initial

    def step(self):
        """Update the learning rate for the current step."""
        self.current_episode += 1
        # just count

    def get_lr(self):
        """Get the current learning rate."""
        return self.current_lr
    
class LRSchedulerExponential:
    def __init__(self, optimizer, lr_initial, lr_final, total_episodes=500):
        """
        Custom Exponential Learning Rate Scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be updated.
            initial_lr (float): Initial learning rate at the beginning of training.
            final_lr (float): Final learning rate at the end of training.
            total_steps (int): Total number of steps (or episodes) during training.
        """
        self.total_episodes = total_episodes
        self.optimizer = optimizer
        self.lr_initial = lr_initial
        self.lr_final = lr_final
        self.decay_rate = (lr_final / lr_initial) ** (1 / self.total_episodes)  # Compute the decay rate
        self.current_episode = 0
        self.current_lr = lr_initial

    def step(self):
        """Update the learning rate for the current step."""
        if self.current_episode < self.total_episodes:
            self.current_lr = max(self.lr_final, self.lr_initial * (self.decay_rate ** self.current_episode))
            for param_group in self.optimizer.param_groups:
                # Update learning rate exponentially
                param_group['lr'] = self.current_lr
        self.current_episode += 1

    def get_lr(self):
        """Get the current learning rate."""
        return self.current_lr

class LRSchedulerLinear:
    def __init__(self, optimizer, lr_initial, lr_final, total_episodes=500):
        """
        Custom Exponential Learning Rate Scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be updated.
            initial_lr (float): Initial learning rate at the beginning of training.
            final_lr (float): Final learning rate at the end of training.
            total_steps (int): Total number of steps (or episodes) during training.
        """
        self.total_episodes = total_episodes
        self.optimizer = optimizer
        self.lr_initial = lr_initial
        self.lr_final = lr_final
        self.decay_rate = (lr_initial - lr_final) / self.total_episodes  # Compute the decay rate
        self.current_episode = 0
        self.current_lr = lr_initial

    def step(self):
        """Update the learning rate for the current step."""
        if self.current_episode < self.total_episodes:
            self.current_lr = max(self.lr_final, self.lr_initial - self.decay_rate * self.current_episode)
            for param_group in self.optimizer.param_groups:
                # Update learning rate exponentially
                param_group['lr'] = self.current_lr
        self.current_episode += 1

    def get_lr(self):
        """Get the current learning rate."""
        return self.current_lr
    
class LRSchedulerDynamic:
    def __init__(self, optimizer, lr_initial, lr_final, total_episodes=500):
        """
        Dynamic Learning Rate Scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be updated.
            initial_lr (float): Initial learning rate at the beginning of training.
            final_lr (float): Final learning rate at the end of training.
            total_steps (int): Total number of steps (or episodes) during training.
        """
        self.total_episodes = total_episodes
        self.optimizer = optimizer
        self.lr_initial = lr_initial
        self.lr_final = lr_final
        self.decay_episodes = int(0.8 * total_episodes)
        self.decay_rate = (lr_initial - lr_final) / self.decay_episodes  # Compute the decay rate
        self.current_episode = 0
        self.current_lr = lr_initial

    def step(self):
        """Update the learning rate for the current step."""
        if self.current_episode < self.decay_episodes:
            self.current_lr = max(self.lr_final, self.lr_initial - self.decay_rate * self.current_episode)
        else:
            self.current_lr = self.lr_final
            
        for param_group in self.optimizer.param_groups:
            # Update learning rate exponentially
            param_group['lr'] = self.current_lr
        self.current_episode += 1

    def get_lr(self):
        """Get the current learning rate."""
        return self.current_lr
    

# For visualization
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))  
    # np.insert(a, 0, 0) adds a 0 at the beginning of a, creating a new array.
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size 
    # Subtracting these slices gives the sum of elements in each moving window of size window_size.
	# Dividing by window_size gives the average for each window.
    #
    # The “tapering effect” at the beginning and end of the array is a way to calculate partial averages 
    # where the moving window cannot fully fit (i.e., at the edges). Since the middle section of the array 
    # allows a full window to compute averages, handling the edges is necessary to avoid leaving gaps in the result. 
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# Trainers
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                #state = env.reset()
                state = env.reset()[0]  # 改！
                done = False
                while not done:
                    action = agent.take_action(state)
                    #next_state, reward, done, _ = env.step(action) 
                    next_state, reward, done, truncated, info = env.step(action)  # 改！
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                