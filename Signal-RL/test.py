# Autorun test script
# Model configurations loaded from the config.yaml file automatically;
# Note: set run_mode, corruption_mode, corruption_ratio, imputation mode to right values before running the script (otherwise environment settings will be incorrect)! 
# test function
def test_agent(env, agent, verbose=False):
    """
    Test the trained agent for one episode.
    
    Args:
        env (gym.Env): The testing environment.
        agent (DQN): The trained agent.
        
    Returns:
        total_reward (float): Total reward obtained in the episode.
    """
    state, _ = env.reset()
    total_reward = 0
    done = False
    step = 0

    while not done:
        # Take an action
        action = agent.take_action(state)
        next_state, reward, done, _, _ = env.step(action)
        # Accumulate reward
        total_reward += reward
        
        # Print step details
        if verbose:
            print(f"Step {step + 1}:")
            print(f"State: {state[:10]}...")  # Show the first 10 elements of the state
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            print(f"Total Reward: {total_reward}")
            print("="*40)
        
        # Move to the next state
        state = next_state
        step += 1

    if verbose:
        print("\nTesting completed!")
        print(f"Total steps: {step}")
        print(f"Total reward: {total_reward}")
    env.close()
    return total_reward


#=====================Main Script=====================
if __name__ == "__main__":
    # Agent testing script
    import torch
    import numpy as np
    import random
    from environment import TrafficSignalEnv
    from tools import get_epsilon_linear, get_epsilon_exponential, get_epsilon_dynamic
    from agent_dqn import GradientDQN, SemiGradientDQN, GradientDoubleDQN, DuelingDQN
    from agent_sarsa import DifferentialSemiGradientSARSA, DifferentialGradientSARSA
    from agent_policy_gradient import ActorCritic, PPO, SAC
    from agent_fixed_time import FixedTimingSignalAgent
    import yaml

    print("starting testing...")
    # Seed for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Map function names as strings to the actual function objects
    epsilon_scheduler_mapping = {
        "get_epsilon_linear": get_epsilon_linear,
        "get_epsilon_exponential": get_epsilon_exponential,
        "get_epsilon_dynamic": get_epsilon_dynamic
    }

    # Map class names as strings to the actual Class
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


    # Load config file
    with open("config.yaml", "r") as file:  
        config = yaml.safe_load(file)
        
    # Hyperparameters
    AgentClassName = config['AgentClassName']
    num_episodes = config["num_episodes"]
    num_parallel_envs = config['num_parallel_envs'] # Set the number of parallel environments
    AgentClass = Agent_mapping[AgentClassName]
    beta = config['beta']  # avg reward update coefficient
    gamma = config['gamma']  # discount factor; Typical Range: 0.9 ~ 0.99;
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

    # Crate env
    env = TrafficSignalEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # Initialize the environment
    env = TrafficSignalEnv()
    env.sumo_binary = "sumo"  # no visualization
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    # Note: agent parameters will be overridden when loading pth file.
    AgentClassName = config['AgentClassName']
    AgentClass = Agent_mapping[AgentClassName]
    if AgentClass == GradientDQN or AgentClass == GradientDoubleDQN or AgentClass == SemiGradientDQN or AgentClass == DuelingDQN:
        agent = AgentClass(state_dim, hidden_dims, action_dim, lr_initial, gamma, epsilon_final, target_update, device)
        on_policy = False
    elif AgentClass == DifferentialSemiGradientSARSA or AgentClass == DifferentialGradientSARSA:
        agent = AgentClass(state_dim, hidden_dims, action_dim, alpha=lr_initial, beta=beta, epsilon=epsilon_initial, device=device)
        on_policy = True
    elif AgentClass == FixedTimingSignalAgent:
        agent = AgentClass()
    else:
        raise Exception("Unkown Agent class")
    
    # load agent
    load_path = config['save_path']
    agent.load(load_path)
    print("Agent loaded successfully!")
    
    # Test the agent
    num_test_episodes = config['num_test_episodes']
    total_rewards = []
    for i in range(num_test_episodes):
        total_reward = test_agent(env, agent, verbose=False)
        total_rewards.append(total_reward)
    print(f"Total rewards from the test episodes:")
    print(total_rewards)
    print(f"Average reward: {np.mean(total_rewards)}")
    print(f"Standard deviation: {np.std(total_rewards)}")
    print("Finished!")

    # Close the environment
    env.close()