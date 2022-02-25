from collections import namedtuple

EnvFeedback = namedtuple('env_feedback', ('state', 'action', 'reward', 'next_state', 'done'))
