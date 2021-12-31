import numpy as np
import pandas as pd
from SingalSeasonal import SignalSeasonal


class AlgoAgent:
    def __init__(self,
                 step_df_rates: pd.DataFrame,
                 config
                 ):
        self.df = step_df_rates
        self.point = config['point_decimal']
        self.time_intervals = [30, 10]
        self.ss_config = config['slow_fast']  # (0.00005, 0.01)
        self.ask_modes = self.get_modes('high')
        self.bid_modes = self.get_modes('low')

        self.ask_z_score = SignalSeasonal(self.ss_config).z_score(self.ask_modes[1], self.ask_modes[0])
        self.bid_z_score = SignalSeasonal(self.ss_config).z_score(self.bid_modes[1], self.bid_modes[0])

    def get_modes(self, price_type: str, ):
        prices = self.df.loc[:, price_type].to_numpy()
        return SignalSeasonal(self.ss_config).trade_modes(prices)

    def _position(self, thr_z=(0.02, 0.02)):
        print(self.ask_z_score[-1] ,  self.ask_z_score[-2])
        if self.ask_z_score[-1] > thr_z[0]:
            print('here')
            if self.ask_z_score[-1] < self.ask_z_score[-2]:
                print('sell zone')
                return "SELL"
        elif self.bid_z_score[-1] < -thr_z[1]:
            if self.ask_z_score[-1] > self.ask_z_score[-2]:
                print('buy zone')
                return "BUY"

    def _price_deal(self):
        alpha_point = 0.0  # self.params['param_price_deal']
        if self._position() == 'SELL':
            print('sell done')
            return self.df.loc[self.df.index[-1], 'close'] + alpha_point
        elif self._position() == 'BUY':
            print('buy done')

            return self.df.loc[self.df.index[-1], 'close'] - alpha_point
        else:
            return False

    def _stop_loss_estimate(self, st_point=50.):
        # calculate stop loss price alpha_point
        if self._position() == 'SELL':
            # st_point = np.mean(abs(self.ask_modes[2] - self.ask_modes[1]))

            return self._price_deal() + 1 * st_point * self.point
        elif self._position() == 'BUY':
            # st_point = np.mean(abs(self.bid_modes[2] - self.bid_modes[1]))

            return self._price_deal() - 1 * st_point * self.point
        else:
            return None

    def _take_profit_estimate(self, tp_point=50):

        # calculate take profit alpha_point
        if self._position() == 'SELL':
            # tp_point = np.mean(abs(self.bid_modes[2] - self.bid_modes[1]))
            return self._price_deal() - tp_point * self.point

        elif self._position() == 'BUY':
            # tp_point = np.mean(abs(self.ask_modes[2] - self.ask_modes[1]))
            return self._price_deal() + tp_point * self.point

        else:
            return None

    def _price_deviation(self):
        # abs(self.ask_modes[-2]-self.bid_modes[-2])
        # calculate price deviation
        return (1 / self.point) * np.mean(abs(self.ask_modes[-2] - self.bid_modes[-2])) / 2

    def _time_expire_order(self, time_delay=2):

        return time_delay * self.time_intervals[-1]

    def _volume_lot(self):

        return 0.01

    def take_action(self, params=(0.03, 0.03, 30, 30, 2)):
        if not self._price_deal():
            # return {"type":"CLOSE"}

            return {}
        return {
            # "action": mt5.TRADE_ACTION_DEAL,
            # "symbol": self.symbol,
            # "volume": self._volume_lot(),
            "type": self._position((params[0], params[1])),
            "price": self._price_deal(),
            "sl": self._stop_loss_estimate(params[2]),
            "tp": self._take_profit_estimate(params[3]),
            "deviation": self._price_deviation(),
            # "magic": 234000,
            # "comment": "python script open",
            # "type_time": ORDER_TIME_SPECIFIED,
            # "type_filling": mt5.ORDER_FILLING_RETURN,
            "expiration": self._time_expire_order(params[4]),
        }


# from FinFeature.OfflineMarket import OfflineMarket
# import matplotlib.pyplot as plt
#
# df = pd.read_csv(f'./ratesM1.csv')
# symbol = 'EURUSD.c'
# market = OfflineMarket(df)
#
# s_config = {
#     "window": (1 * 24 * 60),
#     "freq_modes": [4, 15, 60, 240, int((1 * 12 * 60) / 2 - 1)],
#     "point_decimal": 0.00001,
#     "slow_fast": (0.000005, 0.01)
# }
# tx = 1 * s_config['window']
# for t in range(1 * s_config['window'], 2 * s_config['window']):
#     df = market.ohlcv(symbol).iloc[(t - s_config['window']):t, :]
#     # print(df)
#     # breakpoint()
#     # plt.plot(df.loc[:, 'close'].to_numpy())
#
#     algo = AlgoAgent(df, s_config)
#     print(algo.take_action())
    # p_high = algo.ask_modes
    # p_low = algo.bid_modes
    # plt.plot((df.loc[:, 'high'].to_numpy()))
    # plt.plot((df.loc[:, 'close'].to_numpy()))
    # plt.plot((df.loc[:, 'low'].to_numpy()))
    #
    # for i in range(len(p_low)):
    #     plt.plot((p_high[i]))
    #     plt.plot((p_low[i]))
    #     # plt.ylim([-60, 60])
    #     # plt.xlim([1300, len(df)])
    # plt.show()
    # plt.show(block=False)
    # plt.pause(1.5)
    # plt.close()

#
# #
# print(algo.take_action())
# fig, axs = plt.subplots(5)
#
# for i in range(1,len(algo.ask_modes)):
#
#     axs[i-1].plot(algo.ask_modes[i] - algo.ask_modes[i-1])
#     axs[i-1].axhline(np.mean(abs(algo.ask_modes[i] - algo.ask_modes[i-1])))
#
#     axs[i-1].axhline(-np.mean(abs(algo.bid_modes[i] - algo.bid_modes[i-1])))
#
# axs[4].plot(algo.ask_modes[-1])
# plt.show()
