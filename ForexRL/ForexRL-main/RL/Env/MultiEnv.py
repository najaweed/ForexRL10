import numpy as np
import pandas as pd
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from RL.Env.MonoEnv import mono_env_maker
import torch
import math

def multi_config_maker(market):
    multi_env_conf = {"num_agents": market.num_symbols}
    list_mono_config = []
    for i in range(market.num_symbols):
        state_obs = np.hstack((market.prices()[:, i:(i + 1)],))

        # state_obs = np.array(
        #     torch.randn_like(torch.from_numpy(state_obs), requires_grad=False).cpu())
        state_rew = market.price_point_diff('close')[:, i]
        list_mono_config.append({"state_observations": state_obs, "state_rewards": state_rew})
    multi_env_conf["mono_config"] = list_mono_config
    return multi_env_conf


class MultiEnv(MultiAgentEnv):
    def __init__(self,
                 c_multi_env_config,
                 c_mono_env_maker=mono_env_maker,
                 ):
        self.num_env_agents = c_multi_env_config["num_agents"]
        self.config_mono_env = c_multi_env_config["mono_config"]
        self.env_agents = [c_mono_env_maker(self.config_mono_env[i]) for i in range(self.num_env_agents)]

        self.dones = set()
        self.observation_space = self.env_agents[0].observation_space
        self.action_space = self.env_agents[0].action_space

        self.index_last_step = self.env_agents[0].index_last_step
        self.total_reward = 0.
        self.total_rand_reward = 0.
        self.max_reward = 0.
        self.index_step = 0.

    def reset(self):
        self.dones = set()
        return {i: a.reset() for i, a in enumerate(self.env_agents)}

    def step(self, action_dict):
        # print(action_dict)
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.env_agents[i].step(action)
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.env_agents)
        self.index_step += 1.

        for r in rew.values():
            self.total_reward += r

        step_max_reward = []
        for rand_rew in info.values():
            self.total_rand_reward += rand_rew[0]
            step_max_reward.append(abs(rand_rew[1]))
        self.max_reward += sum(step_max_reward)/len(step_max_reward)
        # self.render()
        return obs, rew, done, info

    def render(self, mode=None):
        return print('\t step:', int(self.index_step),
                     "\t|",
                     "\t max reward:", int(self.max_reward),
                     "\t|",
                     "\t total reward:", int(self.total_reward),
                     "\t|",
                     "\t rand reward:", int(self.total_rand_reward))


#### REGISTER MODEL ####
from FinFeature.OfflineMarket import OfflineMarket

df = pd.read_csv(f'/home/z/Desktop/backups/memory_backup/DB/rates.csv')
market_1 = OfflineMarket(df)


def multi_env_maker(p_multi_env_config):
    return MultiEnv(p_multi_env_config)


multi_env_config = multi_config_maker(market_1)
tune.register_env('Forex', multi_env_maker)

# TODO QMix and grouping agents
# groups = {"G1": [str(i) for i in range(14)],
#           "G2": [str(i) for i in range(14,market_1.num_symbols)]}
#
# def multi_g_env_maker(p_multi_env_config):
#     env = MultiEnv(p_multi_env_config)
#     act_space = gym.spaces.Box(low=-np.float(0.01), high=np.float(0.01), shape=(market_1.num_symbols, 1))
#     obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(market_1.num_symbols,100))
#     return env.with_agent_groups(groups,obs_space ,act_space)
#
# tune.register_env('gForex', multi_g_env_maker)
