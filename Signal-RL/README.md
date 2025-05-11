# Signal-RL Experiment
This folder contains the code and running instructions for the Signal-RL experiment.

![Signal-RL](../assets/sumo_illustration.png "SUMO env illustration")


## Model
DQN model is used for the Signal-RL experiment.

Target network and reply buffer are used for the Signal-RL experiment.

SUMO simulation software is used for the building the environment.

Gym interface is used for the environment.

### Files
- ```environment.py```: contains Environment class, implementing the Gym interface;
- ```tools.py```: some useful functions;
- ```agent.py```: agent class, implementing the RL algorithms and a fixed timing signal;
- ```train.py```: training script;
- ```test.py```: testing script;
- ```config.yaml```: configuration file;
- ```junction.net.xml```: SUMO network file;
- ```junction.rou.xml```: SUMO route file;
- ```junction.sumocfg```: SUMO sumo config file;
- ```agent.pth```: trained model path.


## Training configurations
### Table 2: Signal-RL Configuration

| Task                     | Signal-RL                                                                                          |
|--------------------------|----------------------------------------------------------------------------------------------------|
| **Model type**           | DQN                                                                                                |
| **Architecture & model description** | - `hidden_dims`: [256, 48] <br> - State: road cell occupancy  <br> - Action: next phase for next 6 seconds  <br> - Action dim: 4 <br> - Step reward: *rₜ = -(qₜ - 80)/80*  <br> - Stop speed threshold: 0.3 m/s |
| **Datasets**             | - Single intersection simulation  <br> (1 Episode = 1h = 3600 steps)                               |
| **Training config**      | - `num_episodes`: 50  <br> - `batch_size`: 256  <br> - LR scheduler: linear decay with plateau  <br> - `lr_init`: 1e-3, `lr_final`: 1e-4  <br> - Epsilon scheduler: linear decay with plateau  <br> - `epsilon_init`: 1.0, `epsilon_final`: 1e-2  <br> - Discounting γ: 0.98  <br> - `target_update`: 10  <br> - `buffer_size`: 10000 |


## How to run
Set the configuration in ```config.yaml```;

Run ```train.py``` and ```test.py``` to train and test the model.

Or, you can run ```run_tasks.sh```.