import sys

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
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

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.visit_count = Counter()
        self.success_count = Counter()

        for i in range(self.args.n_allgoals):
            self.visit_count.update(count_list[i])
            self.success_count.update(count_list[i])


    def setup(self, scheme0, scheme1, groups, preprocess0, preprocess1, mac0,mac1):
        self.new_batch = partial(EpisodeBatch, scheme0, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess0, device=self.args.device)
        self.new_tsg_batch = partial(EpisodeBatch, scheme1, groups, self.batch_size, self.episode_limit + 1,
                                      preprocess=preprocess1, device=self.args.device)
        self.agent_mac = mac0
        self.tsg_mac = mac1

        self.learner = None

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.tsg_batch = self.new_tsg_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0

        self.agent_mac.init_hidden(batch_size=self.batch_size)
        self.tsg_mac.init_hidden(batch_size=self.batch_size)
        max_distance = float(min(self.env.max_distance_x,self.env.max_distance_y)) / 2

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)
            pre_tsg_transition_data = {
                "state": [self.env.get_state()],
                "obs": [self.env.get_obs()],
            }
            self.tsg_batch.update(pre_tsg_transition_data,ts = self.t)
            if self.t==0:
                tsg_goals = self.tsg_mac.select_goal(self.tsg_batch,t_ep=self.t,t_env=self.t_env, test_mode=test_mode)
                tsg_delta_t = self.tsg_mac.select_skip(self.tsg_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            else:
                num = len(tsg_delta_t[0])
                new_tsg_goals = self.tsg_mac.select_goal(self.tsg_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                new_tsg_delta_t = self.tsg_mac.select_skip(self.tsg_batch, t_ep=self.t, t_env=self.t_env,test_mode=test_mode)

                for i in range(num):
                    tsg_delta_t[0][i] = tsg_delta_t[0][i]-1
                    if tsg_delta_t[0][i] == -1:
                        tsg_delta_t[0][i] = new_tsg_delta_t[0][i]
                        tsg_goals[0][i] = new_tsg_goals[0][i]
            cpu_tsg_goals = tsg_goals.to("cpu").numpy()
            cpu_tsg_delta_t = tsg_delta_t.to("cpu").numpy()
            pre_transition_data = {
                "subgoals": cpu_tsg_goals,
                "delta_ts":cpu_tsg_delta_t
            }
            self.batch.update(pre_transition_data,ts=self.t)
            actions = self.agent_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()
            reward, terminated, env_info = self.env.step(actions[0])

            if self.args.reward_positive:
                if reward < 0:
                    reward = 0  # reward only positive
            episode_return += reward
            post_transition_data = {
                "reward": [(reward,)],
                "actions": cpu_actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)]
            }
            self.batch.update(post_transition_data, ts=self.t)
            tsg_post_transition_data = {
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "reward": [(reward,)],
                "subgoals": tsg_goals,
                "delta_ts": tsg_delta_t
            }
            self.tsg_batch.update(tsg_post_transition_data,ts = self.t)

            hind_goal = tsg_goals.clone().detach()
            detach_tsg_goals_ = tsg_goals.clone().detach().cpu()
            if self.args.add_goal:
                detach_tsg_goals = detach_tsg_goals_.clone()
                for i in range(self.args.n_agents):
                    detach_tsg_goals[0][i] = detach_tsg_goals_[0][i] + self.args.add_goal_num
            else:
                detach_tsg_goals = detach_tsg_goals_.clone()
            detach_tsg_delta_t = tsg_delta_t.clone().detach().cpu()
            intrinsic_reward = 0
            x_ = self.args.x_length / self.args.x_range
            y_ = self.args.y_length / self.args.y_range
            for i, item in self.env.agents.items():
                pos_x = item.pos.x - self.args.x_start
                pos_y = item.pos.y - self.args.y_start
                if pos_x < 0:
                    pos_x = 0
                if pos_x > self.args.x_length:
                    pos_x = self.args.x_length - 0.01
                if pos_y < 0:
                    pos_y = 0
                if pos_y > self.args.y_length:
                    pos_y = self.args.y_length - 0.01
                x = pos_x / x_
                y = pos_y / y_
                if self.args.reward_int:
                    x = int(x)
                    y = int(y)
                if x == self.args.x_range:
                    x = x - 1
                if y == self.args.y_range:
                    y = y - 1
                hind_goal[0][i] = x + y * self.args.x_range
                goal_x = detach_tsg_goals[0][i] % self.args.x_range
                goal_y = detach_tsg_goals[0][i] / self.args.x_range
                if self.args.hindsight:
                    count_goal = hind_goal[0][i]
                else:
                    count_goal = detach_tsg_goals[0][i]
                self.visit_count.update(count_list[count_goal.item()])
                if reward >= self.args.success_sig:
                    success = True
                else:
                    success = False
                if success:
                    if self.args.success_hindsight:
                        count_success = hind_goal[0][i]
                    else:
                        count_success = detach_tsg_goals[0][i]
                    self.success_count.update(count_list[count_success.item()])
                int_reward = abs(goal_x - x) + abs(goal_y - y)
                intrinsic = np.sqrt(detach_tsg_delta_t[0][i] + 1) / (int_reward + 1) - self.args.r_baseline
                if actions[0][i] == 0:
                    pass
                    # intrinsic = 0
                intrinsic_reward = intrinsic_reward + intrinsic
            H = np.array(list(self.visit_count.values()))
            sum = H.sum()
            H = H / sum
            entropy = - H * np.log(H)
            sum_entropy = entropy.sum()
            # intrinsic_reward = intrinsic_reward / self.args.n_agents
            post_transition_data = {
                "int_reward": [(intrinsic_reward,)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            post_transition_data = {
                "hindsight_goals": hind_goal,
            }
            self.batch.update(post_transition_data, ts=self.t)
            post_transition_data = {
                "hindsight_goals": hind_goal,
                "entropy_reward": [(sum_entropy,)]
            }
            self.tsg_batch.update(post_transition_data, ts=self.t)
            self.t += 1
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1


        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],

        }
        tsg_last_data = {
            "state": [self.env.get_state()],
            "obs": [self.env.get_obs()],

        }
        self.batch.update(last_data, ts=self.t)
        self.tsg_batch.update(tsg_last_data, ts=self.t)

        # Select actions in the last stored state
        last_tsg_goals = self.tsg_mac.select_goal(self.tsg_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        last_tsg_delta_t = self.tsg_mac.select_skip(self.tsg_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        cpu_last_tsg_goals = last_tsg_goals.to("cpu").numpy()
        cpu_last_tsg_delta_t = last_tsg_delta_t.to("cpu").numpy()
        pre_transition_data = {
            "subgoals": cpu_last_tsg_goals,
            "delta_ts": cpu_last_tsg_delta_t,
            "hindsight_goals": cpu_last_tsg_goals
        }
        self.batch.update(pre_transition_data, ts=self.t)
        self.tsg_batch.update(pre_transition_data, ts=self.t)
        # Fix memory leak
        actions = self.agent_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns

        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        self.flag = 0

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.agent_mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.agent_mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, self.tsg_batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
