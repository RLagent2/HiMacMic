from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np


class TSGlevel1MAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.goal_selector = action_REGISTRY[args.action_selector](args)
        self.skip_selector = action_REGISTRY[args.action_selector](args)

        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None
        self.goal_hidden_states = None

    def select_skip(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        #avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs ,_= self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_skips = self.skip_selector.select_action(agent_outputs[bs], None, t_env,
                                                            test_mode=test_mode)
        return chosen_skips

    def select_goal(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        #avail_actions = ep_batch["avail_actions"][:, t_ep]
        _,agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_goals = self.goal_selector.select_action(agent_outputs[bs], None, t_env,
                                                            test_mode=test_mode)
        return chosen_goals

    def forward(self, ep_batch, t, test_mode=False):

        agent_inputs = self._build_inputs(ep_batch, t)
        skip_outs,goal_outs,self.hidden_states = self.tsg1_agent(agent_inputs, self.hidden_states)

        return skip_outs,goal_outs

    def goal_forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        goal_outs ,self.goal_hidden_states = self.tsg1_agent.forward(agent_inputs, self.goal_hidden_states)

        return goal_outs

    def init_hidden(self, batch_size):
        self.hidden_states = self.tsg1_agent.init_hidden()

        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.skip_hidden_states = self.hidden_states

    def parameters(self):
        return self.tsg1_agent.parameters()

    #def skip_parameters(self):
    #    return self.skip_agent.parameters()

    def load_state(self, other_mac):
        self.tsg1_agent.load_state_dict(other_mac.tsg1_agent.state_dict())
        #self.skip_agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.tsg1_agent.cuda()
        #self.skip_agent.cuda()

    def save_models(self, path):
        th.save(self.tsg1_agent.state_dict(), "{}/tsg1_agent.th".format(path))

    def load_models(self, path):
        self.tsg1_agent.load_state_dict(th.load("{}/tsg1_agent.th".format(path), map_location=lambda storage, loc: storage))
        #self.skip_agent.load_state_dict(th.load("{}/skip_agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.tsg1_agent = agent_REGISTRY[self.args.tsg_agent](input_shape, self.args)
        #self.skip_agent = agent_REGISTRY[self.args.skip_agent](input_shape, self.args)


    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["subgoals_onehot"][:, t]))
                inputs.append(th.zeros_like(batch["delta_ts_onehot"][:, t]))
            else:
                inputs.append(batch["subgoals_onehot"][:, t - 1])
                inputs.append(batch["delta_ts_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["subgoals_onehot"]["vshape"][0]
            input_shape += scheme["delta_ts_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)

        return chosen_actions

    def forward_old(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)

        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
