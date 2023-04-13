from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
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


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
                            for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.visit_count = Counter()
        self.success_count = Counter()
        self.Flag = 0

        for i in range(self.args.n_allgoals):
            self.visit_count.update(count_list[i])
            self.success_count.update(count_list[i])

        self.log_train_stats_t = -100000

    def setup(self, scheme0, scheme1, groups, preprocess0, preprocess1, mac0, mac1):
        self.new_batch = partial(EpisodeBatch, scheme0, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess0, device=self.args.device)
        self.new_tsg_batch = partial(EpisodeBatch, scheme1, groups, self.batch_size, self.episode_limit + 1,
                                     preprocess=preprocess1, device=self.args.device)

        self.agent_mac = mac0
        # self.agent_mac.cpu()
        self.tsg_mac = mac1
        # self.tsg_mac.cpu()
        self.scheme0 = scheme0
        self.scheme1 = scheme1
        self.groups = groups
        self.preprocess0 = preprocess0
        self.preprocess1 = preprocess1

        self.learner = None

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()
        self.tsg_batch = self.new_tsg_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        pre_tsg_transition_data = {
            "state": [],
            #    "avail_actions": [],
            "obs": [],
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_tsg_transition_data["state"].append(data["state"])
            pre_tsg_transition_data["obs"].append(data["obs"])
        #    pre_tsg_transition_data["avail_actions"].append(data["avail_actions"])

        self.batch.update(pre_transition_data, ts=0)
        self.tsg_batch.update(pre_tsg_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.agent_mac.init_hidden(batch_size=self.batch_size)
        self.tsg_mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            if self.t == 0:
                tsg_goals = self.tsg_mac.select_goal(self.tsg_batch, t_ep=self.t, t_env=self.t_env,
                                                     bs=envs_not_terminated, test_mode=test_mode)
                tsg_delta_t = self.tsg_mac.select_skip(self.tsg_batch, t_ep=self.t, t_env=self.t_env,
                                                       bs=envs_not_terminated, test_mode=test_mode)
                temp_gg = tsg_goals
                temp_tt = tsg_delta_t
                old_tsg_goals = tsg_goals
                old_tsg_delta_t = tsg_delta_t
            else:
                new_tsg_goals = self.tsg_mac.select_goal(self.tsg_batch, t_ep=self.t, t_env=self.t_env,
                                                         bs=envs_not_terminated, test_mode=test_mode)
                new_tsg_delta_t = self.tsg_mac.select_skip(self.tsg_batch, t_ep=self.t, t_env=self.t_env,
                                                           bs=envs_not_terminated, test_mode=test_mode)
                temp_gg = new_tsg_goals
                temp_tt = new_tsg_delta_t
                number = 0
                for j in range(self.args.batch_size_run):
                    if terminated[j]:
                        continue
                    else:
                        num = len(old_tsg_delta_t[j])
                        for i in range(num):
                            old_tsg_delta_t[j][i] = old_tsg_delta_t[j][i] - 1
                            if old_tsg_delta_t[j][i] >= 0:
                                temp_tt[number][i] = old_tsg_delta_t[j][i]
                                temp_gg[number][i] = old_tsg_goals[j][i]
                            else:
                                old_tsg_delta_t[j][i] = new_tsg_delta_t[number][i]
                                old_tsg_goals[j][i] = new_tsg_goals[number][i]
                        number = number + 1

            cpu_tsg_goals = temp_gg.unsqueeze(1).to("cpu").numpy()
            cpu_tsg_delta_t = temp_tt.unsqueeze(1).to("cpu").numpy()
            pre_transition_data = {
                "subgoals": cpu_tsg_goals,
                "delta_ts": cpu_tsg_delta_t
            }
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.tsg_batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.agent_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated,
                                                    test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()
            cpu_actions2 = actions.unsqueeze(1).to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": cpu_actions2
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }
            pre_transition_data2 = {
                "state": [],
                # "avail_actions": [],
                "obs": []
            }
            post_transition_data2 = {
                "int_reward": [],
            }

            hind_goal = temp_gg.clone().detach()
            detach_tsg_goals_ = temp_gg.clone().detach().cpu()
            detach_tsg_delta_t = temp_tt.clone().detach().cpu()
            temp_goal1 = hind_goal.clone().detach()
            lenth = 0
            for tmd in range(len(terminated)):
                if not terminated[tmd]:
                    lenth = lenth + 1
            temp_goal = temp_goal1[0:lenth, :]

            # Receive data back for each unterminated env
            idd = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    pos = data["pos"]
                    if self.args.add_goal:
                        detach_tsg_goals = detach_tsg_goals_.clone()
                        for i in range(self.args.n_agents):
                            detach_tsg_goals[idd][i] = detach_tsg_goals_[idd][i] + self.args.add_goal_num
                    else:
                        detach_tsg_goals = detach_tsg_goals_.clone()
                    intrinsic_reward = 0
                    x_ = self.args.x_length / self.args.x_range
                    y_ = self.args.y_length / self.args.y_range
                    for i in range(self.args.n_agents):
                        pos_x = pos[i][0] - self.args.x_start
                        pos_y = pos[i][1] - self.args.y_start
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
                            x = x-1
                        if y == self.args.y_range:
                            y = y-1
                        temp_goal[idd][i] = x + y * self.args.x_range
                        goal_x = detach_tsg_goals[idd][i] % self.args.x_range
                        goal_y = detach_tsg_goals[idd][i] / self.args.x_range
                        int_reward = abs(goal_x - x) + abs(goal_y - y)
                        intrinsic = np.sqrt(detach_tsg_delta_t[idd][i] + 1) / (int_reward + 1) - self.args.r_baseline
                        if actions[idd][i] == 0:
                            pass
                            # intrinsic = 0
                        intrinsic_reward = intrinsic_reward + intrinsic

                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))
                    post_transition_data2["int_reward"].append((intrinsic_reward,))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data2["state"].append(data["state"])
                    # pre_transition_data2["avail_actions"].append(data["avail_actions"])
                    pre_transition_data2["obs"].append(data["obs"])
                    idd = idd + 1

            post_transition_data3 = {
                "hindsight_goals": temp_goal.unsqueeze(1).cpu()
            }

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            self.batch.update(post_transition_data2, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.batch.update(post_transition_data3, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.tsg_batch.update(post_transition_data3, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)
            self.tsg_batch.update(pre_transition_data2, bs=envs_not_terminated, ts=self.t, mark_filled=True)


        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
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
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            pos = env.get_pos()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info,
                "pos":pos
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

