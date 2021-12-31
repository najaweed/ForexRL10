import ray
from ray import tune
from ray.rllib.agents import a3c, ppo, pg, dqn
from ray.rllib.policy.policy import PolicySpec
from ray.tune import CLIReporter

from RL.Env.MultiEnv import multi_env_config

pols = {f'policy_{i}': PolicySpec() for i in range(28)}
pol_age_map = (lambda agent_id, episode, worker, **kwargs: f"policy_{agent_id}")

# Limit the number of rows.
reporter = CLIReporter(max_progress_rows=4)
# Add a custom metric column, in addition to the default metrics.
# Note that this must be a metric that is returned in your training results.
reporter.add_metric_column("custom_metric")

ray.init()
# pg.PGTrainer,
# a3c.A2CTrainer,
# dqn.DQNTrainer,
# ppo.PPOTrainer
tune.run(a3c.A2CTrainer,
         progress_reporter=reporter,

         config={
             "env": "Forex",
             "env_config": multi_env_config,
             "framework": "torch",
             "num_workers": 6,
             "num_cpus_per_worker": 1,
             "render_env": True,
             "model": {
                 "fcnet_hiddens": [256, 128, 128, 256],
                 "post_fcnet_hiddens": [512, 512],
                 "post_fcnet_activation": "relu",
                 # == LSTM ==
                 # Whether to wrap the model with an LSTM.
                 "use_lstm": True,
                 # Max seq len for training the LSTM, defaults to 20.
                 "max_seq_len": 20,
                 # Size of the LSTM cell.
                 "lstm_cell_size": 256,
                 # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
                 "lstm_use_prev_action": True,
                 # Whether to feed r_{t-1} to LSTM.
                 "lstm_use_prev_reward": True,
                 # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
                 "_time_major": True,
             },
             "lr": 0.001,

             "num_gpus": 1,
             "multiagent": {
                 "policies": pols,
                 "policy_mapping_fn": pol_age_map,
             },
         }
         )
