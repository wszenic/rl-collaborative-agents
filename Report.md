# Report

## Introduction
This report describes the implementation used to solve the reinforcement learning task. The exact details of the problem
are provided in the `README.md` file.

## Neural network architecture
To solve the problem, a MADDPG was used. As described in the paper, the agents share the replay buffer and utilize the same critic network.
The details of the implementation are in the sections below

### Actor Network

This network i responsible for picking the best action in a given state. Therefore, the network takes as input state
vector and outputs action vector. During my experiments I tried a couple of different sizes, but finally settled on the
following:

* Linear layer (state size=24 -> 256 neurons)
* Linear layer (256 neurons -> 128 neurons)
* Linear layer (128 neurons -> output neurons=4)

The final network is therefore larger than the original implementation in the paper (which suggested using 64 as the
network size). Also, I did use `ReLU` and `BatchNorm1d`
layers in training. The only difference I made was not applying batch normalization to the raw input, as it wasn't very 
effective in my case. That being said, my experiments has shown that adding batchnorm while keeping network atchitecture costant
is very beneficial - the network achieve higher goals and the training is more stable

### Critic Network

Critic network was similar to the one described in the paper - but again larger: 

* Linear layer (state size=24 -> 256 neurons)
* Concatenating output with actions vector (256 + 2 = 258)
* Linear layer (258 neurons -> 128 neurons)
* Linear layer (128 neurons -> output neurons=4)

Similar to the Actor Network, the only differences from the paper are smaller size and not using batchnorm on inputs. 
Also, the original paper proposes using *L2 weight decay* for critic network's optimizer, but in my case that didn't provide 
any useful results.

### Prioritized Experiance replay

My project used OpenAI's implementation of the Prioritized Experience Replay buffer, shared between agents. The implementation
used in this project actually was parametrized using `alpha` and `beta` parameters, used to regulate how much the prioritization 
is used. In my case both parameters were set to `0.9`, which means very strong prioritization (max being 1).  
The size of the replay buffer was set to `1e6`, which follows the suggestions in the DDPG paper.

### Training results, experiments

After settling on the number of neurons which are required to solve the problem, the key issue for me became the stability of
the learning process. The graph below shows average rewards for final experiments:

<img src="plots/episode_score.png"/>

As clearly visible in the picture, the average reward of `0.1` is kept for a very long time, which means that the agents were
able to pass a ball once each and then drop it. After ca. 750 epochs, the score started to rise rapidly, to values above 2. 
Still, the learning process seems largely unstable after 800 epochs, which might indicate lack of resiliance to
different input conditions. 

Visual inspection of the playstyle shows that the rackets move rapidly to the left and right 
and hit the ball quite strongly. Perhaps this causes the ball to fall out of bounds and can cause rapid termination of some episodes. 
The results of the learning process can be examined [here](https://app.neptune.ai/wsz/RL-Tenis/e/RLTEN-150/charts)

## Hyperparameters

The hyperparameters are stored in the `src.config.py` file. The final values were:
* Actor learning rate = `1e-3`
* Critic learning rate = `1e-3`
* Gamma = `0.95`
* Tau = `1e-3`
* Number of epochs = `1000`
* Batch size = `48`
* Frequency of updates = `1`


## Ideas for future developments
The project can be further developed by:
* implementing learning on the raw pixel data
* experiment further with the model parameters to find the optimal values, providing more stable learning process -   
perhaps checking if reducing the frequency of updates will improve the stability
* check how the algorithm performs in different environments (this would not boost the model literally, but would
provide intuition regarding parameters and required model complexity)
* check other learning algorithms (such as AlphaZero or MuZero) to compare them in this specific environment