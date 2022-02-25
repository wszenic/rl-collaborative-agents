# Paths
UNITY_ENV_LOCATION = "/Users/wojciech.szenic/PycharmProjects/rl-collaborative-agents/unity_env/Tennis.app"
CHECKPOINT_SAVE_PATH = "./checkpoints/model_checkpoint.pth"

# Env setup
N_AGENTS = 2

# Epochs
MAX_EPOCH = 5000
EPOCHS_WITH_NOISE = 500

# Hyperparams

ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
GAMMA = 0.99
TAU = 1e-3

UPDATE_FREQ = 200

# Network sizes
ACTOR_SIZE_1 = 128
ACTOR_SIZE_2 = 128

CRITIC_SIZE_1 = 128
CRITIC_SIZE_2 = 128

# Learning
BATCH_SIZE = 256

# Memory
BUFFER_SIZE = int(1e6)
ALPHA = 0.9   # how much prioritization is used
BETA = 0.9  # To what degree to use importance weights

# Noise
NOISE_STD = 0.01

# Checkpoint
CHECKPOINT_EVERY = 500