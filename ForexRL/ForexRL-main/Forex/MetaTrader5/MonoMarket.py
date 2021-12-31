import MetaTrader5 as mt5
import pandas as pd
import numpy as np


class LiveTicks:
    def __init__(self,
                 c_symbol: str,
                 ):
        self.symbol = c_symbol

    def get_ticks(self, start_datetime, end_datetime):
        all_ticks = pd.DataFrame(mt5.copy_ticks_range(self.symbol, start_datetime, end_datetime, mt5.COPY_TICKS_ALL))
        return all_ticks

    def get_rates(self, window_size: int = 100, time_frame=mt5.TIMEFRAME_M1):
        all_prices = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, time_frame, 0, window_size))
        all_prices['time'] = pd.to_datetime(all_prices['time'], unit='s')
        all_prices = all_prices.set_index('time')
        all_prices = all_prices[~all_prices.index.duplicated(keep='first')]
        all_prices = all_prices.reindex(pd.date_range(all_prices.index[0], all_prices.index[-1], freq='min'))
        all_prices = all_prices.fillna(method='bfill')
        all_prices = all_prices.fillna(method='ffill')

        return all_prices



live = LiveTicks('EURUSD.si')
df = live.get_rates(window_size=300)
print(df)
# last_tick = df.index[-1]
# s_config = {
#     "window": (1 * 24 * 60),
#     "freq_modes": [4, 15, 60, 240, int((1 * 12 * 60) / 2 - 1)],
#     "point_decimal": 0.00001,
#     "slow_fast": (0.0000005, 0.01)
# }
#
# import time
# from AlgoAgent import AlgoAgent
# import matplotlib.pyplot as plt
#
# while True:
#     df = live.get_rates(window_size=24 * 60)
#
#     if last_tick < df.index[-1]:
#         last_tick = df.index[-1]
#         print(last_tick)
#         algo = AlgoAgent(df, s_config)
#         # high_modes = algo.ask_modes
#         # low_modes = algo.bid_modes
#         # plt.plot(df.index, df['close'])
#         # plt.plot(df.index,high_modes[0])
#         # plt.plot(df.index,high_modes[1])
#         # plt.plot(df.index,low_modes[0])
#         # plt.plot(df.index,low_modes[1])
#         # plt.show()
#         #
#         # plt.plot(df.index,)
#         print(algo.take_action())
#
#     time.sleep(10)


from datetime import datetime
import pytz


timezone = pytz.timezone("Etc/UTC")
utc_from = datetime(2021, 10, 10,  hour = 13,tzinfo=timezone)
utc_to = datetime(2021, 12, 11, hour = 13, tzinfo=timezone)
rates = mt5.copy_rates_range("USDJPY", mt5.TIMEFRAME_D1, utc_from, utc_to)

print(rates)