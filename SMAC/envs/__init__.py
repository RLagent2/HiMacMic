from functools import partial
from .starcraft import StarCraft2Env
from .multiagentenv import MultiAgentEnv

import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
}


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join("/home/liuchi/", "StarCraftII"))
