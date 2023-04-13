import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: 把mixer变成modulelist，然后返回的值是qtot，第一维度是哪个batch，第二个维度是时间t，第三个是q值，把他们组成一个list返回,OK
# TODO: 如果每个head不一样的网络结构再想想是不是在这改，有些麻烦。NOT DONE

class Mixer(nn.Module):
    def __init__(self, args, abs=True, heads = 5 ):
        super(Mixer, self).__init__()

        self.args = args
        self.abs = abs
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape))

    #TODO:一样的mixer组成head结构,OK
        self.w1_list = nn.ModuleList()
        self.b1_list = nn.ModuleList()
        self.w2_list = nn.ModuleList()
        self.b2_list = nn.ModuleList()

        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))

        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.embed_dim, 1))

        for i in range(heads):
            self.b1_list.append(self.hyper_b1)
            self.w1_list.append(self.hyper_w1)
            self.b2_list.append(self.hyper_b2)
            self.w2_list.append(self.hyper_w2)

    def forward(self, qvals, states):

        out_y = []

        temp_qvals = qvals
        temp_states = states

        for tw1 , tb1,tw2,tb2 in zip(self.w1_list,self.b1_list,self.w2_list,self.b2_list):

            qvals = temp_qvals
            states = temp_states
            # reshape
            b, t, _ = qvals.size()

            qvals = qvals.reshape(b * t, 1, self.n_agents)
            states = states.reshape(-1, self.state_dim)

            # First layer
            w1 = tw1(states).view(-1, self.n_agents, self.embed_dim)  # b * t, n_agents, emb
            b1 = tb1(states).view(-1, 1, self.embed_dim)

            # Second layer
            w2 = tw2(states).view(-1, self.embed_dim, 1)  # b * t, emb, 1
            b2 = tb2(states).view(-1, 1, 1)

            if self.abs:
                w1 = w1.abs()
                w2 = w2.abs()

            # Forward
            hidden = F.elu(th.matmul(qvals, w1) + b1)  # b * t, 1, emb
            y = th.matmul(hidden, w2) + b2  # b * t, 1, 1


            y = y.view(b, t, -1)
            out_y.append(y)


        return out_y

