
from .q_learner import QLearner
from .coma_learner import COMALearner
from .bq_learner import BQLearner
from .tsg_learner import TSGLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["bq_learner"] = BQLearner
REGISTRY["tsg_learner"] = TSGLearner


