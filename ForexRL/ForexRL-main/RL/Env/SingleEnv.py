import gym
import numpy as np
from FinFeature.OfflineMarket import OfflineMarket
import pandas as pd
from ray import tune


class SingleEnv(gym.Env):
    def __init__(self,
                 mono_config,
                 ):
        self.state_observation = mono_config["state_observations"]
        self.state_reward = mono_config["state_rewards"]

        # gym config
        self.action_space = gym.spaces.MultiDiscrete([3 for _ in range(self.state_reward.shape[1])], dtype=int)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_observation.shape[1],))

        # env step parameters
        self.index_current_step: int = 0
        self.index_last_step: int = self.state_observation.shape[0] - 1
        self.total_reward = 0.

    def reset(self):
        self.index_current_step = 0
        self.total_reward = 1.
        return self.state_observation[self.index_current_step, ...]

    def step(self, step_action):
        p_step_action = np.copy(step_action)
        for i, act in enumerate(step_action):
            p_step_action[i] = -1 if act == 2 else act

        step_reward = np.sum(p_step_action * self.state_reward[self.index_current_step])
        self.total_reward += step_reward
        self.index_current_step += 1

        step_observation = self.state_observation[self.index_current_step]
        done = True if self.index_current_step == self.index_last_step else False

        info = {"total_reward": self.total_reward}
        return step_observation, step_reward, done, info

    def render(self, mode="human"):
        print('\ttotal reward', self.total_reward)


df = pd.read_csv(f'/home/z/Desktop/backups/memory_backup/DB/rates.csv')
state_obs = np.hstack((OfflineMarket(df).prices, OfflineMarket(df).log_return))
state_rew = OfflineMarket(df).price_point_diff

single_env_conf = {"state_observations": state_obs, "state_rewards": state_rew}
tune.register_env('SingleForex', SingleEnv)
# test
# env = SingleEnv(single_env_conf)
# for _ in range(10000):
#     print(env.step(env.action_space.sample()))
