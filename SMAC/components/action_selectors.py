import torch
import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
import numpy as np
import math

REGISTRY = {}

class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.skip_schedule = DecayThenFlatSchedule(args.skip_epsilon_start, args.skip_epsilon_finish,
                                                   args.skip_epsilon_anneal_time,
                                                   decay="linear")
        self.epsilon = self.schedule.eval(0)

        self.skip_epsilon = self.skip_schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()

        if avail_actions is not None:
            masked_q_values[avail_actions == 0.0] = -float("inf") # should never be selected!
        else:
            avail_actions = th.ones_like(masked_q_values)
        #masked_q_values[avail_actions == 0] = -float("inf")

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]

        return picked_actions


    def select_skip(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.skip_epsilon = self.skip_schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.skip_epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        # masked_q_values[avail_actions == 0] = -float("inf")  # should never be selected!

        random_numbers = np.random.uniform()

        pick_random = (random_numbers < self.skip_epsilon)
        random_actions = np.random.randint(0, self.args.skip_dim)
        temp = torch.argmax(masked_q_values, dim=2).data.cpu().numpy()[0]
        picked_actions = pick_random * random_actions + (1 - pick_random) * math.floor(np.mean(temp))

        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
