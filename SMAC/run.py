import datetime
import multiprocessing
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.rl_logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

import numpy as np

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)
    # Clean up after finishing
    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")
    # Making sure framework really exits
    os._exit(os.EX_OK)

def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.observation_shape = env_info["obs_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if args.env in ["sc2"]:
        args.n_enemies = env_info["n_enemies"]
        args.obs_ally_feats_size = env_info["obs_ally_feats_size"]
        args.obs_enemy_feats_size = env_info["obs_enemy_feats_size"]
        args.state_ally_feats_size = env_info["state_ally_feats_size"]
        args.state_enemy_feats_size = env_info["state_enemy_feats_size"]
        args.obs_component = env_info["obs_component"]
        args.state_component = env_info["state_component"]
        args.map_type = env_info["map_type"]


    # Default/Base scheme
    agent_scheme_level0 = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        #change
        "int_reward": {"vshape": (1,)},
        "subgoals": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "delta_ts": {"vshape": (1,), "group": "agents", "dtype": th.long},
        #"probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "hindsight_goals": {"vshape": (1,), "group": "agents", "dtype": th.long},
    }
    tsg_scheme_level1 = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "subgoals": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "delta_ts": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "hindsight_goals": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "entropy_reward": {"vshape": (1,)},
        # change
        #"avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        #"probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
    }

    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)]),
        "subgoals": ("subgoals_onehot", [OneHot(out_dim=args.n_subgoals)]),
        "delta_ts": ("delta_ts_onehot", [OneHot(out_dim=args.skip_dim)]),
        "hindsight_goals": ("hindsight_goals_onehot", [OneHot(out_dim=args.n_allgoals)])
    }

    tsg_preprocess = {
        #"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)]),
        "subgoals": ("subgoals_onehot", [OneHot(out_dim=args.n_subgoals)]),
        "delta_ts": ("delta_ts_onehot", [OneHot(out_dim=args.skip_dim)]),
        "hindsight_goals": ("hindsight_goals_onehot", [OneHot(out_dim=args.n_allgoals)])

    }

    agent_buffer_level0 = ReplayBuffer(agent_scheme_level0, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    tsg_buffer_level1 = ReplayBuffer(tsg_scheme_level1, groups, args.tsg_buffer_size, env_info["episode_limit"] + 1,
                          preprocess=tsg_preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    agent_mac_level0 = mac_REGISTRY[args.agent_mac](agent_buffer_level0.scheme, groups, args)
    tsg_mac_level1 = mac_REGISTRY[args.tsg_mac](tsg_buffer_level1.scheme, groups, args)

    # Learner
    learner = le_REGISTRY[args.learner](agent_mac_level0, tsg_mac_level1,
                                        agent_buffer_level0.scheme, logger, args)

    # buffer.scheme is not used, do not care this small bug

    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    runner.setup(scheme0=agent_scheme_level0, scheme1=tsg_scheme_level1, groups=groups,
                 preprocess0=preprocess, preprocess1=tsg_preprocess,
                 mac0=agent_mac_level0, mac1=tsg_mac_level1, )

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))


    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        with th.no_grad():
            for j in range(args.size_run):
                episode_batch, tsg_batch = runner.run(test_mode=False)
                agent_buffer_level0.insert_episode_batch(episode_batch)
                tsg_buffer_level1.insert_episode_batch(tsg_batch)

        if agent_buffer_level0.can_sample(args.batch_size):

            episode_sample = agent_buffer_level0.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        if tsg_buffer_level1.can_sample(args.tsg_sample_size) :
            for p in range(args.mini_batch):
                tsg_sample = tsg_buffer_level1.sample(args.tsg_batch_size)
                # Truncate batch to only filled timesteps
                max_ep_t = tsg_sample.max_t_filled()
                tsg_sample = tsg_sample[:, :max_ep_t]

                if tsg_sample.device != args.device:
                    tsg_sample.to(args.device)

                learner.train_skip(tsg_sample, runner.t_env, episode)
                del tsg_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)

        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)


        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.size_run

        if (runner.t_env- last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env


    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
