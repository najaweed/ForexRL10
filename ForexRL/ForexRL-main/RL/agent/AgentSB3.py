import numpy as np
import pandas as pd
from RL.environment.TestingEnv import TestingEnv
from FinFeature.OfflineMarket import OfflineMarket
from FinFeature.OfflineMarketTrend import OfflineMarketTrend

from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from Forex.MetaTrader5.Symbols import symbols
import time
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
WINDOW_SIZE = 10

df = pd.read_csv(f'\\Users\\Navid\\Desktop\\ForexRL\\ForexRL-main\\FinFeature\\DB\\very_smooths.csv')

mock_market = OfflineMarket(df, symbols)

observ = mock_market.prices

env = TestingEnv(data_observation=observ, window_size=WINDOW_SIZE, market_data_scheme=mock_market)

model = DDPG('MlpPolicy', env , batch_size=500)

s_time = time.time()
# from torchsummary import summary
# print(env.observation_space.shape)
# summary(model.policy, env.observation_space.shape)
time_steps = 50000  # observ.shape[0] - 3000
model.learn(total_timesteps=time_steps )
end_time = time.time()
print(end_time - s_time)
# Save the agent

obs = env.reset()

for i in range(time_steps):
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print('balance', info['balance'])

# plot actions
act = env.action_history
