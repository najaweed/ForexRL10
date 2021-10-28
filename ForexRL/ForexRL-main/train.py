import numpy as np
from RL.environment.TradingEnv import TradingEnv
from Forex.MetaTrader5.Market import MarketData
from Forex.MetaTrader5.Time import MarketTime
from Forex.MetaTrader5.Symbols import Symbols, currencies
from FinFeature.TimeSeries.EMD import E_Modes
from stable_baselines3 import A2C, PPO
import torch

mrk = MarketData(MarketTime().time_range, Symbols(currencies).selected_symbols)

observ = mrk.log_return
env = TradingEnv(data_observation=observ, window_size=30)

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[332, dict(pi=[332, 332], vf=[332, 332])])
# Create the agent
# model = PPO("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)

# train model
# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs)
# model = A2C('MlpPolicy', env)
print('start learning')
model.learn(total_timesteps=8000)
print('end of learning ')
# test model
obs = env.reset()
for i in range(8000):
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(info['balance'])

# plot actions
act = env.action_history
