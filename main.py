import logging
import os

import click
import neptune.new as neptune
import numpy as np
import optuna
from unityagents import UnityEnvironment

from src.agents.maddpg import MADDPGAgent
from src.config.config import settings
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


@run.command("optimize", short_help='Train with hyperparameter optimization')
def optimize():
    study = optuna.create_study(direction='maximize')
    study.optimize(run_with_suggested_settings, n_trials=200, gc_after_trial=True, n_jobs=1)


def run_with_suggested_settings(trial):
    global settings
    settings.epochs_with_noise = trial.suggest_int('epochs_with_noise', 0, 1000)
    settings.actor_learning_rate = trial.suggest_loguniform('actor_learning_rate', 1e-6, 1e-1)
    settings.critic_learning_rate = trial.suggest_loguniform('critic_learning_rate', 1e-6, 1e-1)
    settings.gamma = trial.suggest_float('gamma', 0.5, 0.9999)
    settings.tau = trial.suggest_loguniform('tau', 1e-5, 1e-1)
    settings.update_freq = trial.suggest_int('update_freq', 0, 500)
    settings.actor_size_1 = trial.suggest_int('actor_size_1', 64, 1024)
    settings.actor_size_2 = trial.suggest_int('actor_size_2', 64, 1024)
    settings.critic_size_1 = trial.suggest_int('critic_size_1', 64, 1024)
    settings.critic_size_2 = trial.suggest_int('critic_size_2', 64, 1024)
    settings.batch_size = trial.suggest_int('batch_size', 12, 2048)
    settings.buffer_size = trial.suggest_int('buffer_size', 1e5, 1e8)
    settings.alpha = trial.suggest_float('alpha', float(0), float(1))
    settings.beta = trial.suggest_float('beta', float(0), float(1))
    settings.noise_std = trial.suggest_float('noise_std', float(0), float(1))

    return objective(log=True)


@run.command("train", short_help="Train the reinforcement learning model")
@click.option(
    "-l",
    "--log",
    help="Flag whether the experiment should be logged to neptune.ai",
    required=False
)
def train(log: bool):
    best_score = objective(log)
    print(f"Best score = {best_score}")


def objective(log: bool = True):
    best_average = 0
    env, multi_agent, scores, brain_name = setup_environment(read_saved_model=False, no_graphics=True)
    use_noise = True
    if log:
        neptune_log = log_to_neptune()

    for episode in range(settings.max_epoch):
        if episode > settings.epochs_with_noise:
            use_noise = False
        env_info = env.reset(train_mode=True)[brain_name]
        multi_agent.reset()
        start_state = env_info.vector_observations
        score = act_during_episode(multi_agent, env, start_state, brain_name, use_noise)

        if log:
            neptune_log['best_average'].log(np.mean(scores[-100:]))
            neptune_log['score'].log(score)
        scores.append(score)

        episode_score = np.mean(scores[-100:])

        if episode % 1 == 0:
            print(
                f"Ep: {episode} | Score: {score:.2f} | Max: {np.max(scores):.2f} "
                f"| Avg: {episode_score:.4f}")

        if episode_score > best_average:
            best_average = episode_score
        if episode % settings.checkpoint_every == 0:
            multi_agent.save()

    if log:
        neptune_log.stop()
    multi_agent.save()
    env.close()

    return best_average


def setup_environment(read_saved_model=False, no_graphics=True):
    env = UnityEnvironment(file_name=settings.unity_env_location, no_graphics=no_graphics)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    state = env_info.vector_observations[0]

    multi_agent = MADDPGAgent(state_size=len(state), action_size=brain.vector_action_space_size,
                              read_saved_model=read_saved_model)

    scores = []

    return env, multi_agent, scores, brain_name


def act_during_episode(multi_agent, env, state, brain_name, use_noise):
    score = []
    while True:
        action = multi_agent.act(state, use_noise)

        env_info = env.step(action)[brain_name]

        env_response = EnvFeedback(state, action, env_info.rewards, env_info.vector_observations,
                                   env_info.local_done)

        multi_agent.step(env_response)

        score += [env_response.reward]
        state = env_response.next_state
        if any(env_response.done):
            break
    return np.max(np.sum(np.array(score), axis=0))


def log_to_neptune():
    neptune_run = neptune.init(
        project="wsz/RL-Tenis",
        api_token=os.getenv('NEPTUNE_TOKEN')
    )

    neptune_run['parameters'] = {
        'ACTOR_LEARNING_RATE': settings.actor_learning_rate,
        'CRITIC_LEARNING_RATE': settings.critic_learning_rate,
        'GAMMA': settings.gamma,
        'TAU': settings.tau,
        'BATCH_SIZE': settings.batch_size,
        'ACTOR_MID_1': settings.actor_size_1,
        'ACTOR_MID_2': settings.actor_size_2,
        'CRITIC_CONCAT_1': settings.critic_size_1,
        'CRITIC_CONCAT_2': settings.critic_size_2,
        'EPOCHS_WITH_NOISE': settings.epochs_with_noise,
        'BUFFER_SIZE': settings.buffer_size,
        'ALPHA': settings.alpha,
        'BETA': settings.beta,
        'NOISE_STD': settings.noise_std,
        'UPDATE_FREQ': settings.update_freq
    }
    return neptune_run


if __name__ == "__main__":
    run()
