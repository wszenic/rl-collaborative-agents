import numpy as np
import torch
from torch import nn

from src.config import TAU, CHECKPOINT_SAVE_PATH, \
    NOISE_STD, ACTOR_LEARNING_RATE
from src.network import ActorNet
from src.noise import OUActionNoise
from src.structs import EnvFeedback


def soft_update(local_model, target_model):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)


class DDPG:

    def __init__(self, state_size: int, action_size: int, critic_network: nn.Module, read_saved_model=False,
                 agent_id=None):
        self.noise = OUActionNoise(mean=np.zeros(action_size), std_deviation=float(NOISE_STD) * np.ones(action_size))

        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        self.actor_network_local = ActorNet(state_size, action_size).to(device)
        self.actor_network_target = ActorNet(state_size, action_size).to(device)
        self.actor_network_target.load_state_dict(self.actor_network_local.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor_network_local.parameters(), lr=ACTOR_LEARNING_RATE)

        self.critic_network = critic_network  # in maddpg critic is shared across many agents

        if read_saved_model:
            self.__load_model(agent_id)

        self._step_id = 0

    def __load_model(self, agent_id):
        saved_model = torch.load(CHECKPOINT_SAVE_PATH.format(agent_id=agent_id))
        self.actor_network_local.load_state_dict(saved_model)

    def act(self, state, use_noise: bool):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if use_noise:
            return np.clip(self.__get_state_action_values(state).numpy() + self.noise(), -1, 1)
        else:
            return np.clip(self.__get_state_action_values(state).numpy(), -1, 1)

    def reset(self):
        self.noise.reset()

    def pick_actions(self, next_state: np.array):
        return self.actor_network_target(next_state)

    def learn(self, env_data: EnvFeedback):
        # actor
        actor_actions = self.actor_network_local(env_data.state)
        critic_value = self.critic_network(env_data.state, actor_actions)
        actor_loss = -torch.mean(critic_value)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.actor_network_local, self.actor_network_target)

    def __get_state_action_values(self, state):
        self.actor_network_local.eval()
        with torch.no_grad():
            actions_values = self.actor_network_local(state)
        self.actor_network_local.train()

        return actions_values
