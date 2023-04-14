
# Overcooked

```shell
pip pygame gym
```
The Overcooked scenario is a small-scale game scenario that requires the installation of gym and pygame environments.

Other dependencies：

`pyyaml==5.1`, `tensorboard`, `tensorboardx`, `sacred==0.8.2`, `torch`

## Run HiMacMic


```shell
cd Overcooked
python main.py --config=HiMacMic --env-config=overcooked with env_args.map_name=A
```

Result Visualization

Add `env.render()` to `episode_runner.py` after select action. 

![Alt text](https://github.com/RLagent2/HiMacMic/blob/main/fig/Overcooked.gif)


# GRF

```shell
pip gfootball
```

In Windows system, it's easy to set up the environment following these instructions, but on Linux system, some additional configurations need to be done according to the original repository guidance. Please refer to the official repository for details. 

## Run HiMacMic


```shell
cd GRF
python main.py --config=HiMacMic --env-config=academy_3_vs_1_with_keeper
python main.py --config=HiMacMic --env-config=academy_counterattack_hard 
```

Result Visualization

Due to the requirement of the original repository, the GRF scenes can only be visualized on the Windows platform. Please refer to the official repository for more information about the activity.

![Alt text](https://github.com/RLagent2/HiMacMic/blob/main/fig/GRF.gif)


# SMAC and SMAC-v2

## Installing SMAC

You can install SMAC by using the following command:

```shell
pip install git+https://github.com/oxwhirl/smac.git
pip install git+https://github.com/oxwhirl/smacv2.git
```

Note that if SMAC-v2 is installed over SMAC, some scenarios may encounter issues when running SMAC due to some adjustments made by the original author. For more information, please refer to the official repository.

Alternatively, you can clone the SMAC repository and then install `smac` with its dependencies:

```shell
git clone https://github.com/oxwhirl/smac.git
pip install -e smac/
```
## Installing StarCraft II

SMAC is based on the full game of StarCraft II (versions >= 3.16.1). To install the game, follow the commands bellow. We conducted experiments using version `SC2.4.10 of the game.

### Linux

Please use the Blizzard's [repository](https://github.com/Blizzard/s2client-proto#downloads) to download the Linux version of StarCraft II. By default, the game is expected to be in `~/StarCraftII/` directory. This can be changed by setting the environment variable `SC2PATH`.

### MacOS/Windows

Please install StarCraft II from [Battle.net](https://battle.net). The free [Starter Edition](http://battle.net/sc2/en/legacy-of-the-void/) also works. PySC2 will find the latest binary should you use the default install location. Otherwise, similar to the Linux version, you would need to set the `SC2PATH` environment variable with the correct location of the game.

## SMAC maps

Download the [SMAC Maps](https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip) and extract them to your `$SC2PATH/Maps` directory. If you installed SMAC via git, simply copy the `SMAC_Maps` directory from `smac/env/starcraft2/maps/` into `$SC2PATH/Maps` directory.

After downloading the corresponding map, different experiments can be conducted by adjusting the contents in the config folder during the experiment.

## Run HiMacMic


```shell
cd SMAC
python main.py --config=HiMacMic --env-config=sc2 with env_args.map_name=6h_vs_8z

python main.py --config=HiMacMic --env-config=sc2 with env_args.map_name=2c_vs_64zg

python main.py --config=HiMacMic --env-config=sc2 with env_args.map_name=MMM2
```







