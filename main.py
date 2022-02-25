import logging
import os
import time

import click
import neptune.new as neptune
import numpy as np
from unityagents import UnityEnvironment

from src.agents.maddpg import MADDPGAgent
from src.config import MAX_EPOCH, GAMMA, TAU, BATCH_SIZE, ACTOR_LEARNING_RATE, \
    CRITIC_LEARNING_RATE, CHECKPOINT_EVERY, ACTOR_SIZE_1, CRITIC_SIZE_2, CRITIC_SIZE_1, ACTOR_SIZE_2, \
    UNITY_ENV_LOCATION, EPOCHS_WITH_NOISE, BUFFER_SIZE, ALPHA, BETA, NOISE_STD, UPDATE_FREQ
from src.structs import EnvFeedback


@click.group(chain=True, invoke_without_command=True)
def run():
    logging.info("Running the ml-monitoring project")


@run.command("evaluate", short_help="Run the agent based on saved weights")
def evaluate():
    logging.info("Setting up the environment for evaluation")
    env, multi_agent, scores, brain_name = setup_environment(read_saved_model=True, no_graphics=False)

    env_info = env.reset(train_mode=False)[brain_name]
    start_state = env_info.vector_observations

    score = act_during_episode(multi_agent, env, start_state, brain_name, use_noise=False)
    print(f"Evaluation score = {score}")
    env.close()


@run.command("train", short_help="Train the reinforcement learning model")
@click.option(
    "-l",
    "--log",
    help="Flag whether the experiment should be logged to neptune.ai",
    required=False
)
def train(log: bool):
    env, multi_agent, scores, brain_name = setup_environment()
    use_noise = True
    if log:
        neptune_log = log_to_neptune()

    for episode in range(MAX_EPOCH):
        if episode > EPOCHS_WITH_NOISE:
            use_noise = False
        episode_start_time = time.time()
        env_info = env.reset(train_mode=True)[brain_name]
        multi_agent.reset()
        start_state = env_info.vector_observations
        score = act_during_episode(multi_agent, env, start_state, brain_name, use_noise)

        if log:
            neptune_log['best_average'].log(np.mean(scores[-100:]))
            neptune_log['score'].log(score)
        scores.append(score)
        episode_time = time.time() - episode_start_time
        print(
            f"Ep: {episode} | Score: {score:.2f} | Max: {np.max(scores):.2f} "
            f"| Avg: {np.mean(scores[-100:]):.4f} | Time: {episode_time:.4f}")

        if episode % CHECKPOINT_EVERY == 0:
            multi_agent.save()

    if log:
        neptune_log.stop()
    multi_agent.save()
    env.close()


def setup_environment(read_saved_model=False, no_graphics=True):
    env = UnityEnvironment(file_name=UNITY_ENV_LOCATION, no_graphics=no_graphics)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    state = env_info.vector_observations[0]

    multi_agent = MADDPGAgent(state_size=len(state), action_size=brain.vector_action_space_size,
                              read_saved_model=read_saved_model)

    scores = []

    return env, multi_agent, scores, brain_name


def act_during_episode(multi_agent, env, state, brain_name, use_noise):
    score = 0
    while True:
        action = multi_agent.act(state, use_noise)

        env_info = env.step(action)[brain_name]

        env_response = EnvFeedback(state, action, env_info.rewards, env_info.vector_observations,
                                   env_info.local_done)

        multi_agent.step(env_response)

        score += np.max([env_response.reward, env_response.reward])
        state = env_response.next_state
        if any(env_response.done):
            break
    return score


def log_to_neptune():
    neptune_run = neptune.init(
        project="wsz/RL-Tenis",
        api_token=os.getenv('NEPTUNE_TOKEN')
    )

    neptune_run['parameters'] = {
        'ACTOR_LEARNING_RATE': ACTOR_LEARNING_RATE,
        'CRITIC_LEARNING_RATE': CRITIC_LEARNING_RATE,
        'GAMMA': GAMMA,
        'TAU': TAU,
        'BATCH_SIZE': BATCH_SIZE,
        'ACTOR_MID_1': ACTOR_SIZE_1,
        'ACTOR_MID_2': ACTOR_SIZE_2,
        'CRITIC_CONCAT_1': CRITIC_SIZE_1,
        'CRITIC_CONCAT_2': CRITIC_SIZE_2,
        'EPOCHS_WITH_NOISE': EPOCHS_WITH_NOISE,
        'BUFFER_SIZE': BUFFER_SIZE,
        'ALPHA': ALPHA,
        'BETA': BETA,
        'NOISE_STD': NOISE_STD,
        'UPDATE_FREQ': UPDATE_FREQ
    }
    return neptune_run


if __name__ == "__main__":
    run()
