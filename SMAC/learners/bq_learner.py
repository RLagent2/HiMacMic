import copy
import os

from components.episode_buffer import EpisodeBatch
from modules.mixers.bmix import Mixer
from modules.mixers.vdn import VDNMixer
#from envs.one_step_matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from scipy import stats
from torch.distributions import Categorical


class BQLearner:
    def __init__(self, mac, scheme, logger, args):


        self.args = args
        self.mac = mac
        self.heads = getattr(args,'num_head',5)
        self.p = getattr(args,'p',0.5)
        self.mixer_heads = getattr(args,'mixer_head',2)

        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac[-1].parameters())

        for i in range(self.heads-1):
            self.params += list(mac[i].parameters())

        if  args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "boot_qmix":
            self.mixer = Mixer(args)
        else:
            print("No Mixer!! What are you doing??")
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.train_t = 0

        # th.autograd.set_detect_anomaly(True)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]


        mask_distribution = []

        loss_list = []
        for i in range(self.heads):
            distribution = stats.bernoulli.rvs(self.p)
            mask_distribution.append(distribution)

        for j in range(self.heads):
            temp_mac = self.mac[j]
            # Calculate estimated Q-Values
            mac_out = []
            temp_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = temp_mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
            chosen_action_qvals_ = chosen_action_qvals

            # Calculate the Q-Values necessary for the target
            with th.no_grad():
                temp_target_mac = self.target_mac[j]
                target_mac_out = []
                temp_target_mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    target_agent_outs = temp_target_mac.forward(batch, t=t)
                    target_mac_out.append(target_agent_outs)

                # We don't need the first timesteps Q-Value estimate for calculating targets
                target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

                # Max over target Q-Values/ Double q learning
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])


                if getattr(self.args, 'q_lambda', False):
                    targets = []
                    for i in range(self.mixer_heads):
                        qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                        qvals = self.target_mixer(qvals, batch["state"])
                        temp_qval = qvals[i]
                        temp_target = target_max_qvals[i]

                        target = build_td_lambda_targets(rewards, terminated, mask, temp_target, temp_qval,
                                                         self.args.gamma, self.args.td_lambda)
                        targets.append(target)

                    # targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                    # self.args.gamma, self.args.td_lambda)
                else:
                    targets = []
                    for i in range(self.mixer_heads):
                        temp_target = target_max_qvals[i]

                        target = build_td_lambda_targets(rewards, terminated, mask, temp_target,
                                                         self.args.n_agents, self.args.gamma, self.args.td_lambda)

                        targets.append(target)

            # Mixer
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

            losses = []
            total_loss = 0

            for k in range(self.mixer_heads):
                td_error = (chosen_action_qvals[k] - targets[k].detach())
                td_error = 0.5 * td_error.pow(2)
                mask = mask.expand_as(td_error)
                masked_td_error = td_error * mask
                loss = L_td = masked_td_error.sum() / mask.sum()
                losses.append(loss)
                total_loss = loss

            for kk in range(self.mixer_heads - 1):
                total_loss += losses[kk]
                total_loss = total_loss.mean()

            loss_list.append(total_loss)

        final_total_loss = loss_list[-1]*mask_distribution[-1]

        for j in range(self.heads-1):
            final_total_loss += loss_list[j]*mask_distribution[j]
            final_total_loss = final_total_loss.sum()

        # Optimise
        self.optimiser.zero_grad()
        final_total_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals[0] * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets[0] * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env


    def _update_targets(self):
        for ii in range(self.heads):
            self.target_mac[ii].load_state(self.mac[ii])
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):

        for i in range(self.heads):
            self.mac[i].cuda()

            self.target_mac[i].cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):

        for i in range(self.heads):
            temp_path = os.path.join(path + "_num_" + str(i))
            os.makedirs(temp_path, exist_ok=True)
            self.mac[i].save_models(temp_path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
            #th.save(self.mixer.module.state_dict() if isinstance(self.mixer, th.nn.DataParallel) else self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        #th.save(self.optimiser.module.state_dict() if isinstance(self.optimiser, th.nn.DataParallel) else self.optimiser.state_dict() ,"{}/opt.th".format(path))

    def load_models(self, path):
        for i in range(self.heads):
            temp_path = os.path.join(path + "_num_" + str(i))
            self.mac[i].load_models(temp_path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
