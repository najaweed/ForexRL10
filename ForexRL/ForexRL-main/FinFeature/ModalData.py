import numpy as np
import pandas as pd
from FinFeature.OfflineMarket import OfflineMarket
from Forex.MetaTrader5.Symbols import symbols
from FinFeature.TimeSeries.EMD import imfs, de_trending ,de_fractal , de_noise
import time
import matplotlib.pyplot as plt
from Forex.MetaTrader5.Symbols import symbols
from FinFeature.TimeSeries.Kalman import Kalman_smoother
# df1 = pd.read_csv(f'\\Users\\Navid\\Desktop\\ForexRL\\ForexRL-main\\FinFeature\\DB\\smooths.csv')
# mock_market1 = OfflineMarket(df1, symbols)
# df2 = pd.read_csv(f'\\Users\\Navid\\Desktop\\ForexRL\\ForexRL-main\\FinFeature\\DB\\rates.csv')
# mock_market2 = OfflineMarket(df2, symbols)
# prices = mock_market2.prices[:11111, 0]
#
# start_time = time.time()
# #emd_mode = imfs(prices)
# de_fract4 = Kalman_smoother(prices , Q = 0.001)
# de_fract5 = Kalman_smoother(prices , Q = 0.00000001)

# end_time = time.time()
# print(end_time - start_time)
# #
# plt.plot(mock_market2.prices[:11111, 0], label='raw')
#
# plt.plot(mock_market1.prices[:11111, 0], label='smooth')
#
# plt.plot(de_fract4, label='kalman')
# plt.plot(de_fract5, label='kalman5')
#
# plt.legend()
# plt.show()

print()
class ModalDB:
    def __init__(self,
                 rates_data_frame: pd.DataFrame):
        self.df_rates = rates_data_frame
        self._selecting_symbol = [f'{sym},close' for sym in symbols]
        self.df_trend_rates = pd.DataFrame(columns=self._selecting_symbol)

    def gen_trends(self):
        print(self.df_trend_rates)
        for i , col in enumerate(self.df_rates.columns):
            print(i)
            #if i == 3:
            if col in self._selecting_symbol:

                print(col)
                rates = self.df_rates[col].to_numpy()
                emd_modes  = imfs(rates)
                self.df_trend_rates[col] = de_trending(rates, emd_modes)


        print(self.df_trend_rates)

    def to_csv(self):
        self.gen_trends()
        print(self.df_trend_rates)
        self.df_trend_rates.to_csv('DB\\trends.csv', index=False, )

#ModalDB(mock_market2.all_rates.iloc[:,:]).to_csv()