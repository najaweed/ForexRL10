import random
from ray.rllib.policy.policy import PolicySpec


def gen_policy(i):
    config = {
        "model": {
            "fcnet_hiddens": [512, 512],
            # "post_fcnet_hiddens": [512, 512, 512, 512, 512],
            # "post_fcnet_activation": "relu",
            # == LSTM ==
            # Whether to wrap the model with an LSTM.
            "use_lstm": True,
            # Max seq len for training the LSTM, defaults to 20.
            "max_seq_len": 50,
            # Size of the LSTM cell.
            "lstm_cell_size": 1024,
            # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
            "lstm_use_prev_action": True,
            # Whether to feed r_{t-1} to LSTM.
            "lstm_use_prev_reward": True,
            # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
            "_time_major": True,
        },
        "gamma": random.choice([0.95, 0.99]),
    }
    return PolicySpec(None, config=config, action_space=None, observation_space=None)


policies = {f'policy{i}': gen_policy(i) for i in range(2)}
print(policies)


def policy_agent_map(id_agent, episode, worker, **kwargs):
    return []
