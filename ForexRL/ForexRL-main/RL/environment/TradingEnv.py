import gym
import numpy as np
from RL.environment.Reward import simpleReward
from RL.environment.Action import simpleAction, configAction
from RL.environment.Observ import Observ
from Forex.MetaTrader5.Market import MarketData
from Forex.MetaTrader5.Time import MarketTime


class TradingEnv(gym.Env):

    def __init__(self,
                 data_observation: np.ndarray,
                 window_size: int = 30,
                 market_time_scheme=MarketTime,
                 market_data_scheme=MarketData,
                 action_scheme=simpleAction,
                 reward_scheme=simpleReward,
                 observ_scheme=Observ,
                 ):
        self.window_size = window_size
        self.data_observation = data_observation

        # market schemes
        self.market_time_scheme = market_time_scheme()
        self.market_data_scheme = market_data_scheme(self.market_time_scheme.time_range)
        self.percent_return = self.market_data_scheme.percentage_return
        self.number_agents = self.market_data_scheme.num_symbols

        # environment schemes
        self.action_scheme = action_scheme(configAction(self.number_agents))
        self.reward_scheme = reward_scheme(self.percent_return, self.action_scheme)
        self.observ_scheme = observ_scheme(self.data_observation, self.window_size, self.number_agents)

        # gym spaces
        self.action_space = self.action_scheme.action_space
        self.observation_space = self.observ_scheme.obs_space

        # env episode params
        self.index_end_tick = self.percent_return.shape[0]
        self.index_current_tick = None
        self.action_history = None
        self.reward_history = None
        self.initial_balance = 100000

    def step(self, step_action):
        self.index_current_tick += 1
        step_reward = self.reward_scheme.calculate_reward(step_action,
                                                                      self.index_current_tick,
                                                                      self.action_history,
                                                                      self.initial_balance,
                                                                      )

        self.initial_balance += step_reward
        self.reward_history = np.append(self.reward_history, step_reward)
        self.action_history = np.vstack((self.action_history, step_action))

        done = True if self.index_current_tick == self.index_end_tick else False

        step_observation = self.observ_scheme.provide(self.index_current_tick)
        info = dict(balance=self.initial_balance,
                    )
        return step_observation, step_reward, done, info

    def reset(self):
        self.index_current_tick = self.window_size
        self.reward_history = np.zeros((self.window_size,))
        self.action_history = np.zeros((self.window_size, self.number_agents))
        self.initial_balance = 100000

        return self.observ_scheme.provide(self.index_current_tick)

    def render(self, mode="human"):
        print(self.action_history)
        print(self.reward_history)


