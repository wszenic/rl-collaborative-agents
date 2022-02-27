import json

from pydantic import BaseModel


class ModellerConfig(BaseModel):
    unity_env_location: str
    checkpoint_save_path: str

    n_agents: int
    max_epoch: int
    epochs_with_noise: int
    actor_learning_rate: float
    critic_learning_rate: float
    gamma: float
    tau: float

    update_freq: int

    # network sizes
    actor_size_1: int
    actor_size_2: int

    critic_size_1: int
    critic_size_2: int

    # learning
    batch_size: int

    # memory
    buffer_size: int
    alpha: float  # how much prioritization is used
    beta: float  # to what degree to use importance weights

    # noise
    noise_std: float
    checkpoint_every: int


def read_config_file():
    with open("src/config/config.json", "r") as conf_json:
        config_not_parsed = json.loads(conf_json.read())

    return ModellerConfig(**config_not_parsed)


settings = read_config_file()
