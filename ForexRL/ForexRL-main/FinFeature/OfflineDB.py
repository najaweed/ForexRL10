import pandas as pd

from Forex.MetaTrader5.Market import MarketData
from Forex.MetaTrader5.Time import MarketTime
from Forex.MetaTrader5.Symbols import Symbols, currencies
from FinFeature.TimeSeries.Kalman import Kalman_smoother

mrk = MarketData(MarketTime().time_range, Symbols(currencies).selected_symbols)

df_rates = mrk.all_rates
print(df_rates)
print(df_rates.shape)


class TickDB:
    def __init__(self,
                 rates_data_frame: pd.DataFrame):
        self.df_rates = rates_data_frame
        self.df_smooth_rates = pd.DataFrame(columns=rates_data_frame.columns)
        pass

    def all_smooths(self, Q):
        df_smooth_rates = pd.DataFrame(columns=self.df_rates.columns)

        for i in range(self.df_rates.shape[1]):
            print(i)
            df_smooth_rates.iloc[:, i] = Kalman_smoother(self.df_rates.iloc[:, i].to_numpy(), Q)

        return df_smooth_rates

    def to_csv(self):
        #smooths = self.all_smooths(Q=0.001)
        #very_smooths = self.all_smooths(Q=0.00000001)
        #smooths.to_csv('DB\\smooths.csv', index=False, )
        #very_smooths.to_csv('DB\\very_smooths.csv', index=False, )

        self.df_rates.to_csv('ratesM1.csv', index=False, )

print(TickDB(df_rates).to_csv())
