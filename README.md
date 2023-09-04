# SwarmyMcQLearny
Towards swarm intelligence using deep reinforcement learning

## Overview
SwarmyMcQLearny is a library for deep reinforcement learning method that looks for optimum control of agents. The main objective of this project is to achieve swarm intelligence to solve
a variaty of tasks. The repository will collect DRL algorithms to compare the the performance of the algorithms on different tasks. 
The algorithms are implemented in PyTorch and tensorflow. 
The reason why we use both frameworks is to facilitate the use of the algorithms for the user in case
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
# Try me on colab
to explore and experiment with our SwarmyMcQLearny project in the interactive and user-friendly environment of Google Colab. Just click here
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LuisFMCuriel/SwarmyMcQLearny/blob/main/notebooks/SwarmyMcQLearny.ipynb)

# Getting Started Locally

This guide will walk you through the steps to create a Conda environment named "SwarmyMcQLearny" and install the required packages for your project. Depending on your machine's configuration and your deep learning framework choice (TensorFlow or PyTorch), you'll find instructions below.

## Prerequisites

Before you begin, make sure you have Conda installed. If not, you can download and install it from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

## Create the Conda Environment

Open your terminal and run the following command to create a new Conda environment named "SwarmyMcQLearny":

```bash
conda create --name SwarmyMcQLearny python=3.8
```
Activate the newly created environment using the following command:

```bash
conda activate SwarmyMcQLearny
```
Next, you can install the project requirements using Conda. If you plan to use TensorFlow, use the following command:

```bash
conda install -c anaconda tensorflow-gpu
```

If you prefer PyTorch, run this command:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
Please note that the PyTorch installation command may change over time, and it's essential to check the official PyTorch website for the most up-to-date instructions. Visit [here](https://pytorch.org/get-started/locally/) to verify the installation command for your specific CUDA version.

