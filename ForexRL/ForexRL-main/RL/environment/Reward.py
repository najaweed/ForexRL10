from RL.environment.base.absReward import absReward
from RL.environment.base.absAction import configAction
from RL.environment.Action import simpleAction
import numpy as np


class simpleReward(absReward):
    def __init__(self,
                 percent_return: np.ndarray,
                 action_scheme: simpleAction,
                 ):
        self.percent_return = percent_return
        self.action_scheme = action_scheme
        self.num_agents = self.action_scheme.num_agents
        self.config = self.action_scheme.config

    def calculate_reward(self,
                         action_current_step: np.ndarray,
                         index_current_step: int,
                         action_history: np.ndarray,
                         balance,
                         ):

        if self.config.type_action == 'discrete':

            # {'HOLD': 0, 'BUY': 1, 'SELL': 2}
            step_reward = np.zeros(self.num_agents)
            for i in range(self.num_agents):

                if action_current_step[i] == 0:
                    # closing opened buy position
                    if action_history[index_current_step - 1, i] == 1:
                        step_reward[i] = + self.percent_return[index_current_step, i]
                    # closing opened sell position
                    elif action_history[index_current_step - 1, i] == 2:
                        step_reward[i] = - self.percent_return[index_current_step, i]
                    # holding closed position
                    else:
                        pass
                elif action_current_step[i] == 1:
                    # open new buy position
                    if action_history[index_current_step - 1, i] == 0:
                        step_reward[i] = + self.percent_return[index_current_step, i]
                    # hold buy position
                    elif action_history[index_current_step - 1, i] == 1:
                        step_reward[i] = + self.percent_return[index_current_step, i]
                    elif action_history[index_current_step - 1, i] == 2:
                        # action mask or punish for sequence of buy <-> sell
                        pass
                elif action_current_step[i] == 2:
                    # open new sell  position
                    if action_history[index_current_step - 1, i] == 0:
                        step_reward[i] = - self.percent_return[index_current_step, i]
                    elif action_history[index_current_step - 1, i] == 1:
                        # action mask or punish for sequence of buy <-> sell
                        pass
                    # hold sell position
                    elif action_history[index_current_step - 1, i] == 2:
                        step_reward[i] = - self.percent_return[index_current_step, i]
                else:
                    #
                    pass

            return step_reward

        elif self.config.type_action == 'continuous':
            # TODO add account config such balance/margin and transaction cost
            action_current_step = self.action_scheme.normalize_action(action_current_step)
            action_current_step = action_current_step * balance
            # action_last_step = self.action_scheme.normalize_action(action_history[-1])
            # transaction_cost = self.calculate_transaction(action_current_step,action_last_step )
            percentage_return_portfolio = np.sum(action_current_step * self.percent_return[index_current_step])
            return percentage_return_portfolio# - transaction_cost

    @staticmethod
    def calculate_transaction(action_current_step: np.ndarray,
                              action_last_step: np.ndarray,
                              cost_transaction=0.00001
                              ):
        transactions = 0.
        for i in range(len(action_last_step)):
            if action_last_step[i] == 0. and action_current_step[i] != 0.:
                transactions += abs(action_current_step[i]) * cost_transaction

        return transactions

# from Forex.MetaTrader5.Market import MarketData
# from Forex.MetaTrader5.Time import MarketTime
# from Forex.MetaTrader5.Symbols import Symbols, currencies
# from FinFeature.TimeSeries.EMD import E_Modes
#
# mrk = MarketData(MarketTime().time_range, Symbols(currencies).selected_symbols)
#
# act_config = configAction()
# act_config.number_agents = len(Symbols(currencies).selected_symbols)
# act_config.number_actions = 3
# act_config.type_action = 'continuous'
# act_scheme = simpleAction(act_config)
# # print(act_scheme.action_space)
# rew_scheme = simpleReward(action_scheme=act_scheme, percent_return=mrk.percentage_return[:1000, :])
# step_action = act_scheme.action_space
# act_history = np.random.randint(3, size=(30, 15))
# # print(act_history)
# log_p_history = np.random.rand(30, 15)
# # print(log_p_history)
# for i in range(100):
#     sample_act = rew_scheme.action_scheme.action_space.sample()
#     act_hist = [rew_scheme.action_scheme.action_space.sample(), rew_scheme.action_scheme.action_space.sample()]
#     print(rew_scheme.calculate_reward(sample_act, i, act_hist))
#     np.append(act_hist, sample_act)
