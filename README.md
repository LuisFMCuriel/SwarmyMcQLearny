# SwarmyMcQLearny
Towards swarm intelligence using deep reinforcement learning

## Overview
SwarmyMcQLearny is a library for deep reinforcement learning method that looks for optimum control of agents. The main objective of this project is to achieve swarm intelligence to solve
a variaty of tasks. The repository will collect DRL algorithms to compare the the performance of the algorithms on different tasks. The algorithms are implemented in PyTorch and tenssorflow.
The reason why we use both frameworks is to facilitate the use of the algorithms for the user. The algorithms are implemented in PyTorch and tensorflow. The reason why we use both frameworks is to facilitate the use of the algorithms for the user in case
the user is more familiar with one of the frameworks.

## Library Structure

```
├── LICENSE
├── README.md          <- The top-level README for users of this project.
│
├── requirements.txt   <- The requirements file for reproducing the environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable
├── tensorflow         <- Source code for use in this project with tensorflow framework
│   ├── __init__.py				<- Makes it a Python module
│   │
│   ├── algorithms.py           <- Code to train and test agents implemented in tensorflow
│   │
│   ├── main.py                 <- Code to train and test agents
│   │
│   ├── models.py     	        <- Code to create tensorflow models
│   │
│   ├── replayBuffer.py         <- Code to store experiences and sample from them (Replay Buffer)
│   │
│   ├── strategies.py           <- Code to generate actions for the agents
│   │
│   │
├── pytorch           <- Source code for use in this project with pytorch framework
│   ├── __init__.py				<- Makes it a Python module
│   │
│   ├── algorithms.py           <- Code to train and test agents implemented in pytorch
│   │
│   ├── main.py                 <- Code to train and test agents
│   │
│   ├── models.py     	        <- Code to create pytorch models
│   │
│   ├── replayBuffer.py         <- Code to store experiences and sample from them (Replay Buffer)
│   │
│   ├── strategies.py           <- Code to generate actions for the agents
│   │
├── tests          <- Scripts to test your code
│   ├── __init__.py   
│   ├── test_io_helpers.py   
│   ├── test_pipeline_helpers.py
│   ├── test_training_helpers.py             
│   ├── test_inference_helpers.py             
```

## Getting started
Use the yaml files deposited in SwarmyMcQLearny/environments/ with anaconda to download the python packages needed to run the models. 
Run the next command in a terminal or an Anaconda prompt to create the environment from the yml file.

`conda env create -f environment.yml`

