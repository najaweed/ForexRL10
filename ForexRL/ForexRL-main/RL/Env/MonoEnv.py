import gym
import numpy as np


class MonoEnv(gym.Env):
    def __init__(self,
                 mono_config,
                 ):
        self.state_observation = mono_config["state_observations"]
        self.state_reward = mono_config["state_rewards"]

        # gym config
        # self.action_space = gym.spaces.Box(low=-np.float(.001), high=np.float(.001), shape=(1,), dtype=np.float32)
        self.action_space =gym.spaces.Discrete(3)
        # print('shape observation', self.state_observation.shape)
        if len(self.state_observation.shape) == 1:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        else:
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
        p_step_action = -1 if step_action == 2 else step_action

        step_reward = p_step_action * self.state_reward[self.index_current_step]*np.float32(1e7)
        self.total_reward += step_reward

        self.index_current_step += 1

        step_observation = self.state_observation[self.index_current_step]
        done = True if self.index_current_step == self.index_last_step else False

        info = {"total_reward": self.total_reward}
        # print(self.index_current_step, self.total_reward)

        return step_observation, step_reward, done, info

    def render(self, mode="human"):
        print('\total reward', self.total_reward)


def mono_env_maker(mono_config):
    return MonoEnv(mono_config)

#
# def testing_env():
#     state_obs = np.ones((100, 20))
#     for i in range(state_obs.shape[0]):
#         state_obs[i, ...] *= i
#     state_rew = np.arange(100)
#
#     mono_env = MonoEnv(state_obs, state_rew)
#     obs_0 = mono_env.reset()
#     for _ in range(mono_env.index_last_step):
#         act = np.tanh(np.random.randn())
#         print(mono_env.step(act))


# testing_env()

# def mock_env(MonoEnv):
#
#
# from FinFeature.OfflineMarket import OfflineMarket
# import pandas as pd
#
# df = pd.read_csv(f'/home/z/Desktop/backups/memory_backup/DB/very_smooths.csv')
#
# state_obs = np.hstack((OfflineMarket(df).prices[:, 0:1], OfflineMarket(df).prices[:, 0:1]))
# print(state_obs)
# state_rew = OfflineMarket(df).percentage_return[:, 1]
# #
# mono_conf = {"state_observations": state_obs, "state_rewards": state_rew}
#
# MonoEnv = MonoEnv(mono_conf)
# for _ in range(10):
#     MonoEnv.step(MonoEnv.action_space.sample())

# def test_multi_env():

# import pandas as pd
# from MonoEnv import mono_env_maker
# from FinFeature.OfflineMarket import OfflineMarket
#
#
# df = pd.read_csv(f'/home/z/Desktop/backups/memory_backup/DB/very_smooths.csv')
# market_1 = OfflineMarket(df)
# multi_con = multi_config_maker(market_1)
# multi_env = MultiEnv(mono_env_maker, multi_con)
# multi_env.reset()
# print(multi_env.index_last_step)
# for _ in range(multi_env.index_last_step):
#     act = {i: np.tanh(np.random.randn()) for i in range(markett.num_symbols)}
#
#     print(multi_env.step(act))
