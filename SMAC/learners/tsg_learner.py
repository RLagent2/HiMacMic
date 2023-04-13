import copy
import os
import sys
from components.episode_buffer import EpisodeBatch
from modules.mixers.tsgmix import Mixer
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets,build_td_lambda_targets2
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from scipy import stats
from torch.distributions import Categorical
from collections import Counter

count_list = ['a','b','c','d','e','f',
                'g','h','i','j','k','l',
                    'm','n','o','p','q','r',
                        's','t','u','v','w','x',
                            'y','z','0','1','2','3',
                                '4','5','6','7','8','9',
              'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R',
              'S','T','U','V','W','X','Y','Z',
              '`','~','!','@','#','$','%','^','&','*','(',')','-','_','=','+',
              '[','{',']','}','|',':',';',',','<','>','.','/','?','"'
              ]
class TSGLearner:
    def __init__(self, mac0,mac1, scheme, logger, args):
        self.args = args
        self.agent_mac = mac0
        self.tsg_mac = mac1
        self.logger = logger
        self.last_target_update_episode = 0
        self.last_target_tsg_update_episode = 0
        self.last_save_txt = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.agent_params = list(self.agent_mac.parameters())
        self.tsg_params = list(self.tsg_mac.parameters())

        self.agent_mixer = None
        self.Flag = 1

        self.visit_count = Counter()
        self.success_count = Counter()
        for i in range(self.args.n_allgoals):
            self.visit_count.update(count_list[i])
            self.success_count.update(count_list[i])

        if args.agent_mixer is not None:
            if args.agent_mixer == "vdn":
                self.agent_mixer = VDNMixer()
            elif args.agent_mixer == "tsg_qmix":
                self.agent_mixer = Mixer(args)
            elif args.agent_mixer == "qmix":
                self.agent_mixer = QMixer(args)
            else:
                raise ValueError("No Mixer!! What are you doing??".format(args.agent_mixer))
            self.agent_params += list(self.agent_mixer.parameters())
            self.target_agent_mixer = copy.deepcopy(self.agent_mixer)

        if  args.mixer == "vdn":
            self.tsg_mixer = VDNMixer()
        elif args.mixer == "tsg_qmix":
            self.tsg_mixer = Mixer(args)
        else:
            print("No Mixer!! What are you doing??")

        self.target_tsg_mixer = copy.deepcopy(self.tsg_mixer)
        self.tsg_params += list(self.tsg_mixer.parameters())


        if self.args.optimizer == 'adam':
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
            self.tsg_optimiser = Adam(params=self.tsg_params, lr=args.lr)
        else:
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.tsg_optimiser = RMSprop(params=self.tsg_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_agent_mac = copy.deepcopy(self.agent_mac)
        self.target_tsg_mac = copy.deepcopy(self.tsg_mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

        # th.autograd.set_detect_anomaly(True)

    def train_skip(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        reward = batch["reward"][:, :-1]
#        actions = batch["actions"][:, :-1]
        subgoals = batch["subgoals"][:, :-1]
        delta_ts = batch["delta_ts"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        hindsight_goals = batch["hindsight_goals"][:, :-1]
        entropy_rewards = batch["entropy_reward"][:, :-1]
        rewards = reward + self.args.count_lmd_e * entropy_rewards
#        avail_actions = batch["avail_actions"]
        if reward[0,-1] >= self.args.success_sig:
            success = True
        else:
            success = False
        for i in range(self.args.n_agents):
            if self.args.hindsight:
                count_goal = hindsight_goals[0,:,i]
                if success:
                    count_success = hindsight_goals[0,:,i]
            else:
                count_goal = subgoals[0,:,i]
                if success:
                    count_success = hindsight_goals[0,:,i]

            count_goal = count_goal.squeeze()
            for id in range(count_goal.size(0)):
                self.visit_count.update(count_list[count_goal[id].item()])

            if success:
                count_success = count_success.squeeze()
                for id in range(count_success.size(0)):
                    self.success_count.update(count_list[count_success[id].item()])

        # Calculate estimated Q-Values
        mac_out1 = []
        mac_out2 = []
        self.tsg_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            skip_outs,goal_outs = self.tsg_mac.forward(batch, t=t)
            mac_out1.append(skip_outs)
            mac_out2.append(goal_outs)
        stack_mac_out1 = th.stack(mac_out1, dim=1)  # Concat over time
        stack_mac_out2 = th.stack(mac_out2, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals1 = th.gather(stack_mac_out1[:, :], dim=3, index=delta_ts).squeeze(3)
        #chosen_action_qvals1 = th.gather(mac_out1[:, :-1], dim=3, index=delta_ts).squeeze(3)

        chosen_action_qvals2 = th.gather(stack_mac_out2[:, :], dim=3, index=subgoals).squeeze(3)
        chosen_action_qvals2_2 = th.gather(stack_mac_out2[:, :], dim=3, index=hindsight_goals).squeeze(3)
        chosen_action_qvals_all = stack_mac_out2
        chosen_action_qvals = chosen_action_qvals1+chosen_action_qvals2# Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out1 = []
        target_mac_out2 = []
        self.target_tsg_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_skip_outs ,target_goal_outs = self.target_tsg_mac.forward(batch, t=t)
            target_mac_out1.append(target_skip_outs)
            target_mac_out2.append(target_goal_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        #target_mac_out2 = th.stack(target_mac_out2[1:], dim=1)
        stack_target_mac_out1 = th.stack(target_mac_out1[:], dim=1)  # Concat across time
        stack_target_mac_out2 = th.stack(target_mac_out2[:], dim=1)

        target_max_qvals1 = stack_target_mac_out1.max(dim=3)[0]
        target_max_qvals2 = stack_target_mac_out2.max(dim=3)[0]

        target_max_qvals = target_max_qvals1 + target_max_qvals2

        detach_chosen_action_qvals = chosen_action_qvals.clone().detach()
        detach_target_max_qvals = target_max_qvals.clone().detach()
        detach_chosen_action_qvals_all = chosen_action_qvals_all.clone().detach()
        for i in range(self.args.n_agents):
            final_update_chosen_action_qvals = []
            final_update_target_max_qvals = []
            final_update_state = []
            final_update_reward = []
            final_mask = []
            final_terminated = []
            final_goals = []
            fina_goal_distribution = []
            start = 0
            end = 0
            reward = 0
            for j in range(batch.max_seq_length):
                update_chosen_action_qvals = []
                update_target_max_qvals = []
                update_goal_distribution = []
                update_goal = []
                if j==0:
                    update_chosen_action_qvals.append(chosen_action_qvals[:,j,i])
                    update_goal_distribution.append(chosen_action_qvals_all[:, j, i])
                    if self.args.hindsight:
                        update_goal.append(batch["hindsight_goals"][:, j, i])
                    else:
                        update_goal.append(batch["subgoals"][:, j, i])
                    for k in range(self.args.n_agents):
                        if k != i:
                            update_chosen_action_qvals.append(detach_chosen_action_qvals[:, j, k])
                            update_goal_distribution.append(detach_chosen_action_qvals_all[:, j, k])
                            if self.args.hindsight:
                                update_goal.append(batch["hindsight_goals"][:, j, i])
                            else:
                                update_goal.append(batch["subgoals"][:, j, i])
                    stack_update_chosen_action_qvals = th.stack(update_chosen_action_qvals, dim=1)
                    stack_update_chosen_action_qvals_all = th.stack(update_goal_distribution, dim=1)
                    stack_update_goal = th.stack(update_goal, dim=1)
                    final_update_chosen_action_qvals.append(stack_update_chosen_action_qvals)
                    fina_goal_distribution.append(stack_update_chosen_action_qvals_all)
                    final_goals.append(stack_update_goal)
                    final_update_state.append(batch["state"][:, j])

                else :
                    if batch["delta_ts"][:,j,i,:]==0 or j==batch.max_seq_length-1-1:

                        if j == batch.max_seq_length - 1 -1:
                            reward = reward + pow(self.args.gamma, end + 1) * rewards[:, j - 1, :]
                            update_target_max_qvals.append(target_max_qvals[:, j, i])
                            if self.args.hindsight:
                                update_goal.append(batch["hindsight_goals"][:, j, i])
                            else:
                                update_goal.append(batch["subgoals"][:, j, i])
                            for k in range(self.args.n_agents):
                                if k != i:
                                    update_target_max_qvals.append(detach_target_max_qvals[:, j, k])
                                    if self.args.hindsight:
                                        update_goal.append(batch["hindsight_goals"][:, j, i])
                                    else:
                                        update_goal.append(batch["subgoals"][:, j, i])
                            stack_update_target_max_qvals = th.stack(update_target_max_qvals, dim=1)
                            stack_update_goal = th.stack(update_goal, dim=1)
                            final_update_target_max_qvals.append(stack_update_target_max_qvals)
                            final_goals.append(stack_update_goal)
                            final_update_state.append(batch["state"][:, j])
                            final_update_reward.append(reward)
                            final_terminated.append(batch["terminated"][:, j])
                            break
                        else:
                            reward = reward + pow(self.args.gamma, end + 1) * rewards[:, j - 1, :]
                            end = 0
                            j = j + 1
                            update_chosen_action_qvals.append(chosen_action_qvals[:, j, i])
                            update_target_max_qvals.append(target_max_qvals[:, j, i])
                            update_goal_distribution.append(chosen_action_qvals_all[:, j, i])
                            if self.args.hindsight:
                                update_goal.append(batch["hindsight_goals"][:, j, i])
                            else:
                                update_goal.append(batch["subgoals"][:, j, i])
                            for k in range(self.args.n_agents):
                                if k != i:
                                    update_chosen_action_qvals.append(detach_chosen_action_qvals[:, j, k])
                                    update_goal_distribution.append(detach_chosen_action_qvals_all[:, j, k])
                                    update_target_max_qvals.append(detach_target_max_qvals[:, j, k])
                                    if self.args.hindsight:
                                        update_goal.append(batch["hindsight_goals"][:, j, i])
                                    else:
                                        update_goal.append(batch["subgoals"][:, j, i])
                            stack_update_chosen_action_qvals = th.stack(update_chosen_action_qvals, dim=1)
                            stack_update_chosen_action_qvals_all = th.stack(update_goal_distribution, dim=1)
                            stack_update_target_max_qvals = th.stack(update_target_max_qvals, dim=1)
                            stack_update_goal = th.stack(update_goal, dim=1)
                            final_update_chosen_action_qvals.append(stack_update_chosen_action_qvals)
                            fina_goal_distribution.append(stack_update_chosen_action_qvals_all)
                            final_update_target_max_qvals.append(stack_update_target_max_qvals)
                            final_goals.append(stack_update_goal)
                            final_update_state.append(batch["state"][:, j])
                            final_update_reward.append(reward)
                            final_terminated.append(batch["terminated"][:, j])
                            reward = 0

                    else:
                        reward = reward + pow(self.args.gamma,end+1)*rewards[:,j-1,:]
                        end = end + 1

            stack_final_update_reward = th.stack(final_update_reward,dim=1)
            stack_final_update_chosen_action_qvals =th.stack(final_update_chosen_action_qvals,dim=1)
            stack_final_update_chosen_action_qvals_all = th.stack(fina_goal_distribution, dim=1)
            stack_final_update_target_max_qvals = th.stack(final_update_target_max_qvals,dim=1)
            stack_final_update_state = th.stack(final_update_state,dim=1)
            stack_final_terminated = th.stack(final_terminated,dim=1)
            stack_final_goals = th.stack(final_goals, dim=1)
            H = np.array(list(self.visit_count.values()))
            sum = H.sum()
            H = H / sum
            entropy = - H * np.log(H)
            sum_entropy = entropy.sum()
            S = np.array(list(self.success_count.values()))
            sum = S.sum()
            S = S / sum
            S = th.tensor(S)
            if th.cuda.is_available() and self.args.use_cuda:
                S = S.cuda()
            loss_success = []
            stack_final_goals = stack_final_goals.squeeze(dim=0)[0:-1, :, :]
            for q_values,goal in zip(stack_final_update_chosen_action_qvals_all.squeeze(),stack_final_goals):
                if th.cuda.is_available() == False or self.args.use_cuda == False:
                    q_values = q_values.cpu()
                for id in range(self.args.n_agents):
                    if id == i:
                        if q_values.size(0) != self.args.n_agents or q_values.size(1) != self.args.goal_dim:
                            map = q_values.squeeze()
                        else:
                            map = q_values[id, :].squeeze()
                        p_exp = th.exp(map)
                        p_sum = th.sum(p_exp)
                        pi_exp = p_exp / p_sum
                        goal_id = goal[id, :].squeeze()
                        cross_loss = -S[goal_id] * th.log(pi_exp)
                        cross_loss = cross_loss.sum() / self.args.goal_dim
            #loss_success = th.tensor(loss_success)
            if sys.platform == 'linux' and self.args.use_cuda:
                # loss_success = loss_success.cuda()
                cross_loss = cross_loss.cuda()


            mix_chosen_action_qvals = self.tsg_mixer(stack_final_update_chosen_action_qvals, stack_final_update_state[:, :-1])
            mix_target_max_qvals = self.target_tsg_mixer(stack_final_update_target_max_qvals, stack_final_update_state[:, 1:])


            targetss = stack_final_update_reward + self.args.gamma * (1 - stack_final_terminated) * mix_target_max_qvals
            targets = build_td_lambda_targets2(stack_final_update_reward, stack_final_terminated, mask,
                                               mix_target_max_qvals,
                                               self.args.n_agents, self.args.gamma, self.args.td_lambda2)


            td_error = (mix_chosen_action_qvals - targets.detach())
            td_error = 0.5 * td_error.pow(2)
            loss = td_error + cross_loss * self.args.count_lmd
            loss = loss.sum() / td_error.size(1)
            if i == 0:
                total_loss = loss
            else:
                total_loss = total_loss + loss
            #mask = mask.expand_as(td_error)
            # 0-out the targets that came from padded data
            #masked_td_error = td_error * mask

        self.tsg_optimiser.zero_grad()
        th.autograd.set_detect_anomaly = True
        total_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.tsg_params, self.args.grad_norm_clip)
        self.tsg_optimiser.step()





        if (episode_num - self.last_target_tsg_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_tsg_targets()
            self.last_target_tsg_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("macro_loss", loss.item(), t_env)



    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # Get the relevant quantities
        reward = batch["reward"][:, :-1]
        #int_reward = self.calc_intrinsic_reward(batch)
        int_reward = batch["int_reward"][:, :-1]

        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        #subgoals = batch["subgoals"][:, :-1]
        #delta_ts = batch["delta_ts"][:, :-1]
        rewards = reward + self.args.beta * int_reward
        mac_out = []
        self.agent_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.agent_mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            target_mac_out = []
            self.target_agent_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_agent_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_agent_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_agent_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                                 self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                                  self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qvals = self.agent_mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())

        td_error = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = L_td = masked_td_error.sum() / mask.sum()

        # Optimise
        self.agent_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_agent_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("agent_loss_td", L_td.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("agent_td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("agent_q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("agent_target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env


    def calc_intrinsic_reward(self,batch: EpisodeBatch):
        states = batch["state"]
        obs = batch["obs"]
        goal_pos = batch["subgoals"]
        delta_t = batch["delta_ts"]

        obs_goal = []
        for i in range(self.args.n_agents):
            pos_obs = obs[:,:,i,0:2]
            obs_goal.append(pos_obs)
        stack_obs = th.stack(obs_goal,dim=2)
        goal_obs = goal_pos.clone().detach()
        goal_obs[:,:,:,0]=stack_obs[:,:,:,0] // (self.args.x_length / self.args.x_range) + ((stack_obs[:,:,:,1]+0.42) // (self.args.y_length / self.args.y_range) ) * self.args.x_range

        int_reward = goal_pos.clone().detach()
        float_int_reward = int_reward.float()
        final_int_reward = float_int_reward.clone().detach()
        int_reward[:, :, :, 0] = 1 + abs(goal_obs[:,:,:,0] // self.args.x_range - goal_pos[:,:,:,0] // self.args.x_range) + abs(goal_obs[:,:,:,0] % self.args.x_range - goal_pos[:,:,:,0] % self.args.x_range)

        delta_t_add = delta_t.clone().detach()
        delta_t_add[:, :, :, 0] = delta_t[:, :, :, 0] + 1.0
        final_int_reward[:, :, :, 0] = th.div(delta_t_add[:, :, :, 0], int_reward[:, :, :, 0] )
        tot_int_reward = th.sum(final_int_reward,dim=2)
        return tot_int_reward

    def _update_agent_targets(self):

        self.target_agent_mac.load_state(self.agent_mac)
        if self.agent_mixer is not None:
            self.target_agent_mixer.load_state_dict(self.agent_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _update_tsg_targets(self):
        self.target_tsg_mac.load_state(self.tsg_mac)
        if self.tsg_mixer is not None:
            self.target_tsg_mixer.load_state_dict(self.tsg_mixer.state_dict())
        self.logger.console_logger.info("Updated target tsg/macro network")

    def cuda(self):

        self.agent_mac.cuda()
        self.target_agent_mac.cuda()
        self.tsg_mac.cuda()
        self.target_tsg_mac.cuda()

        if self.agent_mixer is not None:
            self.agent_mixer.cuda()
            self.target_agent_mixer.cuda()
        if self.tsg_mixer is not None:
            self.tsg_mixer.cuda()
            self.target_tsg_mixer.cuda()

    def save_models(self, path):
        self.agent_mac.save_models(path)
        if self.agent_mixer is not None:
            th.save(self.agent_mixer.state_dict(), "{}/agent_mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        self.tsg_mac.save_models(path)
        if self.tsg_mixer is not None:
            th.save(self.tsg_mixer.state_dict(), "{}/tsg_mixer.th".format(path))
        th.save(self.tsg_optimiser.state_dict(), "{}/tsg_opt.th".format(path))

    def load_models(self, path):
        self.agent_mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_agent_mac.load_models(path)
        if self.agent_mixer is not None:
            self.agent_mixer.load_state_dict(th.load("{}/agent_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))

        self.tsg_mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_tsg_mac.load_models(path)
        if self.tsg_mixer is not None:
            self.tsg_mixer.load_state_dict(
                th.load("{}/tsg_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.tsg_optimiser.load_state_dict(
            th.load("{}/tsg_opt.th".format(path), map_location=lambda storage, loc: storage))
