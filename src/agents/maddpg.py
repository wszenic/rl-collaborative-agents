from collections import namedtuple

import torch

from src.agents.ddpg_agent import DDPG, soft_update
from src.config import N_AGENTS, CRITIC_LEARNING_RATE, ALPHA, UPDATE_FREQ, BATCH_SIZE, BETA, GAMMA, TAU, \
    CHECKPOINT_SAVE_PATH
from src.memory.openai import PrioritizedReplayBuffer
from src.network import CriticNet
from src.structs import EnvFeedback


class MADDPGAgent:

    def __init__(self, state_size: int, action_size: int, read_saved_model=False):

        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        self.critic_local = CriticNet(state_size, action_size).to(device)
        self.critic_target = CriticNet(state_size, action_size).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=CRITIC_LEARNING_RATE,
                                                 weight_decay=0)

        self.agents = [
            DDPG(state_size, action_size, self.critic_local, read_saved_model) for _ in range(N_AGENTS)
        ]

        self.memory = PrioritizedReplayBuffer(action_size, ALPHA)

        self._step_id = 0  # needed to trigger learning after a couple of steps

    def step(self, env_data):
        self.memory.add(
            obs_t=env_data.state,
            action=env_data.action,
            reward=env_data.reward,
            obs_tp1=env_data.next_state,
            done=env_data.done,
        )

        self._step_id += 1

        if self._step_id % UPDATE_FREQ == 0:
            if len(self.memory) > BATCH_SIZE:
                sampled_experience = self.memory.sample(BATCH_SIZE, BETA)

                [agent.learn(EnvFeedback(*observation)) for agent, observation in zip(self.agents, sampled_experience)]

    def act(self, state, use_noise: bool):
        [agent.act(agent_state, use_noise) for agent, agent_state in zip(self.agents, state)]

    def reset(self):
        [agent.reset() for agent in self.agents]

    def learn(self, env_data: EnvFeedback):
        for agent in self.agents:
            target_actions = agent.pick_actions(env_data.next_state)

            y = env_data.reward + GAMMA * self.critic_target(
                env_data.next_state, target_actions
            ) * (1 - env_data.done)

            # critic
            critic_value = self.critic_local(env_data.state, env_data.action)
            critic_loss = torch.nn.functional.mse_loss(y, critic_value)

            self.critic_local.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_local.step()

            soft_update(self.critic_local, self.critic_target)

            agent.learn(env_data)

    def save(self):
        [torch.save(agent.actor_network_local.state_dict(), CHECKPOINT_SAVE_PATH) for agent in self.agents]
