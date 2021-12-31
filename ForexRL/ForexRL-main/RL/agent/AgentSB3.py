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


#### alter

# trainer = pg.PGTrainer(env='Forex',
#
#                          config={
#                              "framework": "torch",
#                              # "num_gpus": 1,
#                              #"num_workers": 10,
#                              #"env_config":multi_env_config,
#                              "model": {
#                                  "fcnet_hiddens": [512, 512],
#                                  # "post_fcnet_hiddens": [512, 512, 512, 512, 512],
#                                  # "post_fcnet_activation": "relu",
#                                  # == LSTM ==
#                                  # Whether to wrap the model with an LSTM.
#                                  "use_lstm": True,
#                                  # Max seq len for training the LSTM, defaults to 20.
#                                  "max_seq_len": 150,
#                                  # Size of the LSTM cell.
#                                  "lstm_cell_size": 1024,
#                                  # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
#                                  "lstm_use_prev_action": True,
#                                  # Whether to feed r_{t-1} to LSTM.
#                                  "lstm_use_prev_reward": True,
#                                  # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
#                                  "_time_major": True,
#                              },
#                              "lr": 0.04,
#
#                              "num_gpus": 1,
#                              "multiagent": {
#                                  "policies": pols,
#                                  "policy_mapping_fn": pol_age_map,
#                              }},
#                          )
# print(trainer)
# print(trainer.get_policy().action_space_struct)

#
# for i in range(1000):
#    # Perform one iteration of training the policy with PPO
#    result = trainer.train()
#    print(pretty_print(result))
#
#    if i % 100 == 0:
#        checkpoint = trainer.save()
#        print("checkpoint saved at", checkpoint)
