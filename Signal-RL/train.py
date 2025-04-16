# Agent training script
import torch
import numpy as np
import random
from tools import make_env, ReplayBuffer, moving_average
from tools import get_epsilon_linear, get_epsilon_exponential, get_epsilon_dynamic
from tools import LRSchedulerConstant, LRSchedulerLinear, LRSchedulerExponential, LRSchedulerDynamic
from agent_dqn import GradientDQN, SemiGradientDQN, GradientDoubleDQN, DuelingDQN
from agent_sarsa import DifferentialSemiGradientSARSA, DifferentialGradientSARSA
from agent_policy_gradient import ActorCritic, PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from environment import TrafficSignalEnv
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml, json

# Map agent class names strings to the actual Class
Agent_mapping = {
    "SemiGradientDQN": SemiGradientDQN,
    "GradientDQN": GradientDQN,
    "GradientDoubleDQN": GradientDoubleDQN,
    "DuelingDQN": DuelingDQN,
    "DifferentialSemiGradientSARSA": DifferentialSemiGradientSARSA,
    "ActorCritic": ActorCritic,
    "PPO": PPO,
    "SAC": SAC
}
# Epsilon (for exploration) scheduler function mapping
epsilon_scheduler_mapping = {
    "get_epsilon_linear": get_epsilon_linear,
    "get_epsilon_exponential": get_epsilon_exponential,
    "get_epsilon_dynamic": get_epsilon_dynamic
}
# Learning rate scheduler class mapping
LRScheduler_mapping = {
    "LRSchedulerConstant": LRSchedulerConstant,
    "LRSchedulerExponential": LRSchedulerExponential,
    "LRSchedulerLinear": LRSchedulerLinear,
    "LRSchedulerDynamic": LRSchedulerDynamic,
}

#================Training Function===============
def train(envs, replay_buffer, agent, num_episodes):
    """
    Trains the DQN agent using vectorized parallel environments.
    """
    return_list = []
    with tqdm(total=num_episodes, desc="Training Progress") as pbar:
        for i_episode in range(num_episodes):
            # Update epsilon for each episode
            agent.epsilon = epsilon_scheduler(i_episode, epsilon_initial=epsilon_initial, epsilon_final=epsilon_final, total_episodes=num_episodes)
            
            if num_parallel_envs == 1:
                # "envs" is actually a single env, not vectorized!
                # only off-policy training procedures provided
                episode_return = 0
                state, _ = envs.reset() # Note the number of elements returned
                done = False
                while True:
                    action = agent.take_action(state)
                    next_state, reward, done, _, __ = envs.step(action) # Note the number of elements returned
                    if done:
                        break
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
            
            else: # more than ones envs
                # Initialize tracking variables
                episode_return = np.zeros(num_parallel_envs)  # Track returns for each environment
                states = envs.reset()
                actions = agent.take_action(states)  # Use \(\epsilon\)-greedy action selection
                # Don't use states, _ = env.reset()!!
                # SB3 will handle correctly!
                # the vectorized environment wrapper (DummyVecEnv or SubprocVecEnv) does not return two values, unlike the base Gym API. It returns just the batched states.
                dones = [False] * num_parallel_envs
                while True:
                    if on_policy == True:  # on-policy, like SARSA
                        next_states, rewards, dones, infos = envs.step(actions)
                        # where the parallelization happens when you’re using SubprocVecEnv (or any vectorized environment in Stable-Baselines3).
                        # The SubprocVecEnv distributes actions across multiple environments, steps them in parallel, and collects results efficiently.
                        # your main process is blocked while waiting for all subprocesses to complete their step() execution.
                        # 
                        # Note:
                        # The Gym API specifies that env.step(action) returns exactly five values:
                        #         observations, rewards, dones, infos = envs.step(actions)
                        # However, when you are using SubprocVecEnv from Stable-Baselines3, 
                        # it expects to handle the environments in parallel and batch these values 
                        # for multiple environments. For a batched environment, the step() 
                        # function only returns four values, as shown in the Stable-Baselines3 API:
                        #         bservations, rewards, dones, infos = envs.step(actions)
                        #  when you’re working with SubprocVecEnv, there is no need to handle the truncated flag separately because it is not explicitly returned by SubprocVecEnv. Instead:
                        # dones will include both done (terminal) and truncated (truncated episodes).
                        
                        if any(dones):
                            # if any env is doen, the rewards maynot be completely collected
                            break
                        
                        # Take action and observe next state, reward, and done signal
                        episode_return += rewards

                        # Choose next action \( A' \) based on policy
                        next_actions = agent.take_action(next_states)

                        # on-policy algorithm uses only real-time experiences
                        # low sample efficiency!
                        transition_dict = {
                            'states': states, 
                            'actions': actions,
                            'next_states': next_states,
                            'rewards': rewards,
                            'next_actions': next_actions,
                            'dones': dones
                        }

                        # Update the agent (SARSA update)
                        agent.update(transition_dict)

                        # Update current state and action
                        states = next_states
                        actions = next_actions
                        
                    else: # off-policy, like Q-learning
                        # Select actions for all environments
                        actions = agent.take_action(states)
                        # Store transitions in replay buffer
                        next_states, rewards, dones, infos = envs.step(actions)
                        
                        if any(dones):
                            # if any env is doen, the rewards maynot be completely collected
                            break
                        
                        # off-policy algorithm use replay buffer and draw samples from buffer to learn
                        # higher sample efficiency
                        # add experiences one by one
                        for i in range(num_parallel_envs):
                            replay_buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
                        
                        # Update states and accumulate rewards
                        states = next_states
                        #episode_return += np.mean(rewards)
                        episode_return += rewards  # Accumulate rewards for all environments    

                        # Train the agent when we have enough data
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {
                                'states': b_s, 
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d
                            }
                            
                            # update Q-learning agent (off-policy)
                            agent.update(transition_dict)
                # Log average return across all environments
                avg_return = np.mean(episode_return)
                return_list.append(avg_return)
                
            # adjust learning rate if needed
            # adjust for each episode
            lr_scheduler.step()

            # episode returns visualition
            #return_list.append(episode_return)
            pbar.set_postfix({'Episode': i_episode + 1,
                            'Epsilon': f'{agent.epsilon:.3f}',
                            'LR': f'{lr_scheduler.current_lr:.6f}',
                            'Avg reward': f'{agent.avg_reward:.3f}' if on_policy == True else 'None',
                            'Episode return': f'{return_list[-1]:.3f}'},)
            pbar.update(1)
            #tqdm.write(f'Episode: {i_episode} Epsilon: {agent.epsilon} Return: {avg_return}')
    return return_list


#================Main Script================
if __name__ == '__main__':
    # Load config file
    with open("config.yaml", "r") as file:  
        config = yaml.safe_load(file)
    # Seed for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    #----------------Hyperparameters--------
    AgentClassName = config['AgentClassName']
    num_episodes = config["num_episodes"]
    num_parallel_envs = config['num_parallel_envs'] # Set the number of parallel environments
    print('AgentClass:', AgentClassName)
    AgentClass = Agent_mapping[AgentClassName]
    beta = config['beta']  # avg reward update coefficient
    gamma = config['gamma']  # discount factor; Typical Range: 0.9 ~ 0.99;
    print("Configurations:")
    print(config)
    # Tasks with Short Time Horizons, Lower values like  \gamma = 0.8  or  \gamma = 0.9  might work better (e.g., cart-pole balancing or other reactive tasks).
    # Tasks with Long Time Horizons, Higher values like  \gamma = 0.95  to  \gamma = 0.99  are generally better for environments where long-term planning is important (e.g., traffic signal optimization, navigation, etc.).
    # Rationale: This encourages the agent to prioritize cumulative rewards (e.g., smooth traffic flow) rather than short-term gains (e.g., clearing just one set of cars).
    batch_size = config['batch_size']
    # Epsilon-greedy exploration schedule
    epsilon_initial = config['epsilon_initial']
    epsilon_final = config['epsilon_final']
    epsilon_scheduler_name = config['epsilon_scheduler_name']
    epsilon_scheduler = epsilon_scheduler_mapping[epsilon_scheduler_name]
    # Traffic signal optimization often benefits from significant exploration early in training since the state space is large and non-intuitive strategies might emerge over time. 
    # Initial \epsilon: 1.0; Final \epsilon: 0.01
    hidden_dims = config['hidden_dims']  # hidden layer sizes
    lr_initial = config["lr_initial"]
    lr_final = config["lr_final"]
    lr_scheduler_name = config['lr_scheduler_name']
    target_update = config['target_update']  # target network update frequency
    buffer_size = config['buffer_size']  # replay buffer size
    minimal_size = config['minimum_size']  # minimum size of replay buffer
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Device:', device)
    #---------------------------------------- 
   
    # Create parallel environments
    if num_parallel_envs > 1:
        envs = SubprocVecEnv([make_env() for i in range(num_parallel_envs)])
        #envs = DummyVecEnv([make_env(seed=i) for i in range(num_parallel_envs)])  # for debugging
    else: # if only one env, use original env
        #envs = DummyVecEnv([make_env(seed=i) for i in range(num_parallel_envs)])
        envs = TrafficSignalEnv()
    state_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.n
    print('state_dim:', state_dim, 'action_dim:', action_dim)
    # state is flattened, shape: (2405,); 
    # Note: although SSB3 is used, the state_dim is still single env's
    
    # Create agent
    if AgentClass == GradientDQN or AgentClass == GradientDoubleDQN or AgentClass == SemiGradientDQN or AgentClass == DuelingDQN:
        agent = AgentClass(state_dim, hidden_dims, action_dim, lr_initial, gamma, epsilon_initial, target_update, device)
        on_policy = False
    elif AgentClass == DifferentialSemiGradientSARSA or AgentClass == DifferentialGradientSARSA:
        agent = AgentClass(state_dim, hidden_dims, action_dim, alpha=lr_initial, beta=beta, epsilon=epsilon_initial, device=device)
        on_policy = True
    else:
        raise Exception("Unknown Agent class")
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(buffer_size)
    
    # Create learning rate scheduler
    LRScheduler = LRScheduler_mapping[lr_scheduler_name]
    lr_scheduler = LRScheduler(agent.optimizer, lr_initial=lr_initial, lr_final=lr_final, total_episodes=num_episodes)
    
    # Start training...
    print('=============Start training...============')
    return_list = train(envs, replay_buffer, agent, num_episodes)
    envs.close()
    
    # Save episode return list
    with open('episode_returns.json', 'w') as file:
        json.dump(return_list, file)
    # Save the agent after training
    save_path = config['save_path']
    agent.save(save_path)
    # Plot the results
    plot_figure = config['plot_figure']
    episode_list = list(range(1, len(return_list) + 1))
    
    # Visualization
    plt.plot(episode_list, return_list, label="Return")
    plt.ylim(-1000, 600)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Isolated traffic signal with DQN RL control')
    plt.savefig("Figure_1.png", format="png", dpi=300)  # dpi adjusts the resolution
    if plot_figure:
        plt.show()
    # moving averaged return
    mv_return = moving_average(return_list, 5)
    plt.plot(episode_list, mv_return, label="Moving Avg Return")
    plt.ylim(-1000, 600)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Isolated traffic signal with DQN RL control')
    plt.legend()
    plt.savefig("Figure_2.png", format="png", dpi=300)  # dpi adjusts the resolution
    if plot_figure:
        plt.show()