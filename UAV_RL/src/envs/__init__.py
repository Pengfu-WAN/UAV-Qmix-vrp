from functools import partial
from envs.VRPenv import MultiAgentEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["VRP"] = partial(env_fn, env=MultiAgentEnv)