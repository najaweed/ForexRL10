import ray
from ray import tune
from ray.rllib.agents import a3c , impala , ppo, pg
from ray.rllib.policy.policy import PolicySpec

from RL.Env.MultiEnv import multi_env_config

pols = {f'policy_{i}': PolicySpec() for i in range(28)}
pol_age_map = (lambda agent_id, **kwargs: f"policy_{agent_id}")

ray.init()

tune.run(pg.PGTrainer,

         config={
             "env": "Forex",
             "env_config": multi_env_config,
             "framework": "torch",
             "num_workers": 3,
             "num_cpus_per_worker": 1,

             "model": {
                 "fcnet_hiddens": [512, 512],
                 # "post_fcnet_hiddens": [512, 512, 512, 512, 512],
                 # "post_fcnet_activation": "relu",
                 # == LSTM ==
                 # Whether to wrap the model with an LSTM.
                 "use_lstm": True,
                 # Max seq len for training the LSTM, defaults to 20.
                 "max_seq_len": 150,
                 # Size of the LSTM cell.
                 "lstm_cell_size": 1024,
                 # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
                 "lstm_use_prev_action": True,
                 # Whether to feed r_{t-1} to LSTM.
                 "lstm_use_prev_reward": True,
                 # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
                 "_time_major": True,
             },
             "lr": 0.04,

             "num_gpus": 1,
             "multiagent": {
                 "policies": pols,
                 "policy_mapping_fn": pol_age_map,
             },
         }
         )

# trainer = ppo.PPOTrainer(env='Forex',
#
#                          config={
#                              "framework": "torch",
#                              "num_gpus": 1,
#                              "num_workers": 10,
#                              "env_config":multi_con}
#                          )
# print(trainer)
# print(trainer.train())

#
# for i in range(1000):
#    # Perform one iteration of training the policy with PPO
#    result = trainer.train()
#    print(pretty_print(result))
#
#    if i % 100 == 0:
#        checkpoint = trainer.save()
#        print("checkpoint saved at", checkpoint)
