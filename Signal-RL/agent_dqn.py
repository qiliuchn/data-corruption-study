# Agent algorithms
# "Grad"/"SemiGrad": gradient or semi-gradient learning
# "DQN"/"SARSA": Q-learning or SARSA learning
# "Diff": differential learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from tools import ReplayBuffer
import yaml

# Load config file
with open("config.yaml", "r") as file:  
    config = yaml.safe_load(file)
env_step_duration = config['env_step_duration']

# Neural nets for Q-function approximation
class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = torch.nn.Linear(hidden_dims[2], action_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # First hidden layer
        x = F.relu(self.fc1(x))  # Apply Linear → LayerNorm → ReLU
        x = self.dropout(x)
        # Second hidden layer
        x = F.relu(self.fc2(x))  # Apply Linear → LayerNorm → ReLU
        x = self.dropout(x)
        x = F.relu(self.fc3(x))  # Apply Linear → LayerNorm → ReLU
        # Output layer
        x = self.fc4(x)  # No activation function (raw Q-values)
        return x


class QnetShallow(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = torch.nn.Linear(hidden_dims[1], action_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # First hidden layer
        x = F.relu(self.fc1(x))  # Apply Linear → LayerNorm → ReLU
        x = self.dropout(x)
        # Second hidden layer
        x = F.relu(self.fc2(x))  # Apply Linear → LayerNorm → ReLU
        x = self.dropout(x)
        # Output layer
        x = self.fc3(x)  # No activation function (raw Q-values)
        return x


class QnetDeep(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.fc5 = torch.nn.Linear(hidden_dims[3], action_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # First hidden layer
        x = F.relu(self.fc1(x))  # Apply Linear → LayerNorm → ReLU
        x = self.dropout(x)
        # Second hidden layer
        x = F.relu(self.fc2(x))  # Apply Linear → LayerNorm → ReLU
        x = self.dropout(x)
        x = F.relu(self.fc3(x))  # Apply Linear → LayerNorm → ReLU
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        # Output layer
        x = self.fc5(x)  # No activation function (raw Q-values)
        return x    
    

class QnetGelu(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = torch.nn.Linear(hidden_dims[2], action_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # First hidden layer
        x = F.gelu(self.fc1(x))  # Apply Linear → LayerNorm → ReLU
        x = self.dropout(x)
        # Second hidden layer
        x = F.gelu(self.fc2(x))  # Apply Linear → LayerNorm → ReLU
        x = self.dropout(x)
        x = F.gelu(self.fc3(x))  # Apply Linear → LayerNorm → ReLU
        # Output layer
        x = self.fc4(x)  # No activation function (raw Q-values)
        return x    


class QnetLn(nn.Module):
    '''
    Qnet with layer normalization, no dropout
    '''
    def __init__(self, state_dim, hidden_dims, action_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])  # LayerNorm after first hidden layer
        self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.ln2 = nn.LayerNorm(hidden_dims[1])  # LayerNorm after second hidden layer
        self.fc3 = torch.nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = torch.nn.Linear(hidden_dims[2], action_dim)

    def forward(self, x):
        # First hidden layer
        x = F.relu(self.ln1(self.fc1(x)))  # Dimension: hidden_dims[0]
        # Second hidden layer
        x = F.relu(self.ln2(self.fc2(x)))  # Dimension: hidden_dims[1]
        # Third hidden layer
        x = F.relu(self.fc3(x))  # Dimension: hidden_dims[2]
        # Output layer
        x = self.fc4(x)  # Dimension: action_dim
        return x


class QnetLnShallow(nn.Module):
    '''
    Qnet with layer normalization, no dropout
    '''
    def __init__(self, state_dim, hidden_dims, action_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])  # LayerNorm after first hidden layer
        self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = torch.nn.Linear(hidden_dims[1], action_dim)

    def forward(self, x):
        # First hidden layer
        x = F.relu(self.ln1(self.fc1(x)))  # Dimension: hidden_dims[0]
        # Second hidden layer
        x = F.relu(self.fc2(x))  # Dimension: hidden_dims[2]
        # Output layer
        x = self.fc3(x)  # Dimension: action_dim
        return x
    

class QnetLnShallower(nn.Module):
    '''
    Qnet with layer normalization, no dropout
    '''
    def __init__(self, state_dim, hidden_dims, action_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])  # LayerNorm after first hidden layer
        self.fc2 = torch.nn.Linear(hidden_dims[0], action_dim)

    def forward(self, x):
        # First hidden layer
        x = F.relu(self.ln1(self.fc1(x)))  # Dimension: hidden_dims[0]
        # Output layer
        x = self.fc2(x)  # Dimension: action_dim
        return x  

class QnetLnGeluShallow(nn.Module):
    '''
    Qnet with layer normalization, no dropout
    '''
    def __init__(self, state_dim, hidden_dims, action_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])  # LayerNorm after first hidden layer
        self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = torch.nn.Linear(hidden_dims[1], action_dim)

    def forward(self, x):
        # First hidden layer
        x = F.gelu(self.ln1(self.fc1(x)))  # Dimension: hidden_dims[0]
        # Second hidden layer
        x = F.gelu(self.fc2(x))  # Dimension: hidden_dims[2]
        # Output layer
        x = self.fc3(x)  # Dimension: action_dim
        return x

# For Dueling DQN
class VAnet(nn.Module):
    '''Dueling DQN'''
    def __init__(self, state_dim, hidden_dims, action_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc_A = torch.nn.Linear(hidden_dims[2], action_dim)
        self.fc_V = torch.nn.Linear(hidden_dims[2], 1)

    def forward(self, x):
        # First hidden layer
        x = F.relu(self.ln1(self.fc1(x)))
        # Second hidden layer
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.fc3(x))  # Apply Linear → LayerNorm → ReLU
        # Output layer
        v = self.fc_V(x)
        a = self.fc_A(x)
        # Q-value is calculated as v + a - a.mean
        q = v + a - a.mean(1, keepdim=True)
        return q

class VAnetShallow(nn.Module):
    '''Dueling DQN'''
    def __init__(self, state_dim, hidden_dims, action_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc_A = torch.nn.Linear(hidden_dims[1], action_dim)
        self.fc_V = torch.nn.Linear(hidden_dims[1], 1)

    def forward(self, x):
        # First hidden layer
        x = F.relu(self.ln1(self.fc1(x)))
        # Second hidden layer
        x = F.relu(self.fc2(x))
        # Output layer
        v = self.fc_V(x)
        a = self.fc_A(x)
        # Q-value is calculated as v + a - a.mean
        q = v + a - a.mean(1, keepdim=True)
        return q





# DQN agents
Qnet_mapping = {
    "Qnet": Qnet,  # 3 hidden layers; ReLU activation; with dropout
    "QnetShallow": QnetShallow,  # 2 hidden layers; ReLU activation; with dropout
    "QnetDeep": QnetDeep,  # 4 hidden layers; ReLU activation; with dropout
    "QnetGelu": QnetGelu,  # 3 hidden layers; GeLU activation; with dropout
    "QnetLn": QnetLn,  # 3 hidden layers; with LayerNorm
    "QnetLnShallow": QnetLnShallow,  # 2 hidden layers; with LayerNorm
    "QnetLnShallower": QnetLnShallower,  # 1 hidden layer; with LayerNorm
    "QnetLnGeluShallow": QnetLnGeluShallow,  # 1 hidden layer; GeLU activation; with LayerNorm
    "VANet": VAnet,
    "VANetShallow": VAnetShallow
}
NeuralNetClassName = config['NeuralNetClassName']
NeuralNetClass = Qnet_mapping[NeuralNetClassName]

class GradientDQN:
    '''
    Cf. HORL, p76
    '''
    def __init__(self, state_dim, hidden_dims, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = NeuralNetClass(state_dim, hidden_dims, action_dim).to(device)
        print('Qnet:', self.q_net.__class__)
        print(f'Model size: {sum(p.numel() for p in self.q_net.parameters()) / 1e6:.3f}M')
        # target network
        self.target_q_net = NeuralNetClass(state_dim, hidden_dims, action_dim).to(device)
        # optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update  # target update frequency
        self.count = 0  # count for target update
        self.device = device
    
    def take_action(self, state):
        """
        Takes an action given a state or batch of states.
        
        Args:
            state (np.ndarray or list): Single state (shape: [state_dim]) or batch of states (shape: [batch_size, state_dim])
        
        Returns:
            np.ndarray or int: Single action (if input is single state) or batch of actions (if input is batch of states)
        """
        if isinstance(state, np.ndarray) and len(state.shape) == 1:
            # Single state
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                state = torch.tensor([state], dtype=torch.float).to(self.device)  # Add batch dimension
                action = self.q_net(state).argmax(dim=1).item()  # Remove batch dimension
        elif isinstance(state, np.ndarray) and len(state.shape) == 2:
            # Batch of states
            random_actions = np.random.randint(self.action_dim, size=state.shape[0])
            q_values = self.q_net(torch.tensor(state, dtype=torch.float).to(self.device))
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
            action = np.where(np.random.random(size=state.shape[0]) < self.epsilon, random_actions, greedy_actions)
        else:
            raise ValueError("State must be a numpy array of shape (state_dim,) or (batch_size, state_dim)")

        return action
    
    def update(self, transition_dict):
        '''
        The Q-learning update equation should not discount the future (max_next_q_values) using (1 - dones) 
        because the end of the episode is not caused by the environment dynamics.
	    Artificial termination can lead to instability or bias because the agent doesn’t learn a smooth continuation across episode boundaries
        '''
        # transition_dict is a dict of minibatch data
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # tensor.max(1) returns a tuple of (values, indices); '[0]' selects the values
        q_targets = rewards + self.gamma * max_next_q_values
        
        # Don't add* (1 - dones)
        #loss = torch.mean(F.mse_loss(q_values, q_targets))
        loss = F.smooth_l1_loss(q_values, q_targets)  # Huber loss for more robust training:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # target network update
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    def save(self, path):
        """
        Save the model, optimizer, and other parameters to the specified path.
        """
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'count': self.count,
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path):
        """
        Load the model, optimizer, and other parameters from the specified path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.gamma = checkpoint['gamma']
        self.count = checkpoint['count']
        print(f"Model loaded from {path}")
        
        
class GradientDoubleDQN(GradientDQN):
    def update(self, transition_dict):
        '''
        The Q-learning update equation should not discount the future (max_next_q_values) using (1 - dones) 
        because the end of the episode is not caused by the environment dynamics.
	    Artificial termination can lead to instability or bias because the agent doesn’t learn a smooth continuation across episode boundaries
        '''
        # transition_dict is a dict of minibatch data
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        # -------------changes made to DQN-------------
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)  # Note: double DQN, use q_net to obtain next action
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)  # use target_q_net to obtain q values
        # ----------------------------------------------
        # tensor.max(1) returns a tuple of (values, indices); '[0]' selects the values
        q_targets = rewards + self.gamma * max_next_q_values
        
        # Don't add* (1 - dones)
        #loss = torch.mean(F.mse_loss(q_values, q_targets))
        loss = F.smooth_l1_loss(q_values, q_targets)  # Huber loss for more robust training:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # target network update
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        
        
class SemiGradientDQN(GradientDQN):    
    def update(self, transition_dict):
        # transition_dict is a dict of minibatch data
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # tensor.max(1) returns a tuple of (values, indices); '[0]' selects the values
        # Don't add* (1 - dones)
        # ------------changes made to DQN-------------
        td_errors = rewards + self.gamma * max_next_q_values - q_values
        # Normalize the TD errors to ensure they don’t grow excessively large, which can cause numerical instability
        td_errors = td_errors.detach()
        #td_errors = td_errors.clamp(-1, 1)
        td_errors = td_errors / (1 + td_errors.abs())
        loss = - (td_errors.detach() * q_values).mean()
        # ---------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # target network update
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


# Dueling DQN
# Cf. HORL, p88
class DuelingDQN:
    def __init__(self, state_dim, hidden_dims, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.q_net = NeuralNetClass(state_dim, hidden_dims, self.action_dim).to(device)
        self.target_q_net = NeuralNetClass(state_dim, hidden_dims, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def take_action(self, state):
        """
        Takes an action given a state or batch of states.
         - VAnet
            - LayerNorm
         - Double net
         - Huber loss
        
        Args:
            state (np.ndarray or list): Single state (shape: [state_dim]) or batch of states (shape: [batch_size, state_dim])
        
        Returns:
            np.ndarray or int: Single action (if input is single state) or batch of actions (if input is batch of states)
        """
        if isinstance(state, np.ndarray) and len(state.shape) == 1:
            # Single state
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                state = torch.tensor([state], dtype=torch.float).to(self.device)  # Add batch dimension
                action = self.q_net(state).argmax(dim=1).item()  # Remove batch dimension
        elif isinstance(state, np.ndarray) and len(state.shape) == 2:
            # Batch of states
            random_actions = np.random.randint(self.action_dim, size=state.shape[0])
            q_values = self.q_net(torch.tensor(state, dtype=torch.float).to(self.device))
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
            action = np.where(np.random.random(size=state.shape[0]) < self.epsilon, random_actions, greedy_actions)
        else:
            raise ValueError("State must be a numpy array of shape (state_dim,) or (batch_size, state_dim)")

        return action
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)  # Note: double DQN, use q_net to obtain next action
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)  # use target_q_net to obtain q values
        q_targets = rewards + self.gamma * max_next_q_values
        # no (1 - dones)
        #loss = torch.mean(F.mse_loss(q_values, q_targets))
        loss = F.smooth_l1_loss(q_values, q_targets)  # Huber loss for more robust training:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    def save(self, path):
        """
        Save the model, optimizer, and other parameters to the specified path.
        """
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'count': self.count,
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path):
        """
        Load the model, optimizer, and other parameters from the specified path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.gamma = checkpoint['gamma']
        self.count = checkpoint['count']
        print(f"Model loaded from {path}")
        
        
        
        

#================Training Function===============
# for test only
def train(env, replay_buffer, agent, num_episodes):
    return_list = []
    for i in range(10):  # 分十次打印
        with tqdm(total=num_episodes//10, desc='Iteration %d' %i) as pbar:
            for i_episode in range(num_episodes // 10):
                episode_return = 0
                state, _ = env.reset() # Note the number of elements returned
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _, __ = env.step(action) # Note the number of elements returned
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer数量超过一定值后才开始训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s, 
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 1 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes//10 * i + i_episode + 1),
                                    'return': '%.3f' % np.mean(return_list[-1])})
                pbar.update(1)
    return return_list
            
#================Main Script========================
if __name__ == "__main__":
    lr = 1e-3
    num_episodes = 30
    hidden_dims = [128, 32, 8] # hidden layer sizes
    gamma = 0.98  # discount factor
    epsilon = 0.01  # small probability for exploration
    target_update = 10  # target network update frequency
    buffer_size = 10000  # replay buffer size
    minimal_size = 500  # minimum size of replay buffer
    batch_size = 64
    
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    #device = torch.device('cpu')
         
    from environment import TrafficSignalEnv
    env = TrafficSignalEnv()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print('state_dim:', state_dim, 'action_dim:', action_dim)
    agent = GradientDQN(state_dim, hidden_dims, action_dim, lr, gamma, epsilon, target_update, device)
    #agent = DifferentialSemiGradientSARSA(state_dim, hidden_dims, action_dim, alpha=lr, beta=0.001, epsilon=epsilon, device=device)
    return_list = train(env, replay_buffer, agent, num_episodes)
    save_path = config['save_path']
    agent.save(save_path)
    
    # plot
    from tools import moving_average
    import matplotlib.pyplot as plt
    import json
    
    # Save episode_returns
    with open('episode_returns.json', 'w') as file:
        json.dump(return_list, file)
    
    # Visualization
    episode_list = list(range(1, len(return_list) + 1))
    plt.plot(episode_list, return_list)
    plt.ylim(-600, 100)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Isolated traffic singal with DQN RL control')
    plt.savefig("Figure_1.png", format="png", dpi=300)  # dpi adjusts the resolution
    #plt.show()
    # moving average
    mv_return = moving_average(return_list, 9)
    with open('episode_returns_smoothed.json', 'w') as file:
        json.dump(return_list, file)
    plt.plot(episode_list, mv_return)
    plt.ylim(-600, 100)
    plt.xlabel('Episodes')
    plt.ylabel('Returns (MA)')
    plt.title('Isolated traffic singal with DQN RL control')
    plt.savefig("Figure_2.png", format="png", dpi=300)  # dpi adjusts the resolution
    #plt.show()