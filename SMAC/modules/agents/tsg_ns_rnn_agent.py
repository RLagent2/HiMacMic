import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.agents.tsg_rnn_agent import TSGRNNAgent

class TSGNSRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TSGNSRNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList([TSGRNNAgent(input_shape, args) for _ in range(self.n_agents)])

    def init_hidden(self):
        # make hidden states on same device as model
        return th.cat([a.init_hidden() for a in self.agents])

    def forward(self, inputs, hidden_state):
        hiddens = []
        goals = []
        skips = []

        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                skip, goal,h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i])
                hiddens.append(h)
                skips.append(skip)
                goals.append(goal)
            return th.cat(skips),th.cat(goals), th.cat(hiddens).unsqueeze(0)
        else:
            for i in range(self.n_agents):
                inputs = inputs.view(-1, self.n_agents, self.input_shape)
                skip, goal,h = self.agents[i](inputs[:, i], hidden_state[:, i])

                skips.append(skip.unsqueeze(1))
                goals.append(goal.unsqueeze(1))
                hiddens.append(h.unsqueeze(1))
            out_q_skip = th.cat(skips, dim=1)
            return th.cat(skips, dim=1),th.cat(goals, dim=1),th.cat(hiddens, dim=1)

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)


