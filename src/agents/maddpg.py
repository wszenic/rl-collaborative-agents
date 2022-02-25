import numpy as np
import torch

from src.agents.ddpg_agent import DDPG, soft_update
from src.config import N_AGENTS, CRITIC_LEARNING_RATE, ALPHA, UPDATE_FREQ, BATCH_SIZE, BETA, GAMMA, \
    CHECKPOINT_SAVE_PATH, BUFFER_SIZE
from src.memory.openai import PrioritizedReplayBuffer
from src.network import CriticNet
from src.structs import EnvFeedback


class MADDPGAgent:

    def __init__(self, state_size: int, action_size: int, read_saved_model=False):
        self.action_size = action_size

        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        self.critic_local = CriticNet(state_size, action_size).to(device)
        self.critic_target = CriticNet(state_size, action_size).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=CRITIC_LEARNING_RATE,
                                                 weight_decay=0)

        self.agents = [
            DDPG(state_size, action_size, self.critic_local, read_saved_model, agent_id=i) for i in range(N_AGENTS)
        ]

        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, ALPHA)

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

                self.learn(sampled_experience)

    def act(self, state, use_noise: bool):
        agents_actions = [agent.act(agent_state, use_noise) for agent, agent_state in zip(self.agents, state)]
        return np.array(agents_actions).reshape((len(self.agents), self.action_size))

    def reset(self):
        [agent.reset() for agent in self.agents]

    def learn(self, env_feedback: tuple):
        for agent_id, agent in enumerate(self.agents):
            env_data = EnvFeedback(*[x[:, agent_id] for x in env_feedback])

            target_actions = agent.pick_actions(env_data.next_state)

            y = env_data.reward + GAMMA * self.critic_target(
                env_data.next_state, target_actions
            ).detach().numpy() * (1 - env_data.done)
            y = torch.Tensor(y)

            # critic
            critic_value = self.critic_local(env_data.state, env_data.action)
            critic_loss = torch.nn.functional.mse_loss(y, critic_value)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

            soft_update(self.critic_local, self.critic_target)

            agent.learn(env_data)

    def save(self):
        [torch.save(agent.actor_network_local.state_dict(), CHECKPOINT_SAVE_PATH.format(agent_id=id)) for id, agent in
         enumerate(self.agents)]
