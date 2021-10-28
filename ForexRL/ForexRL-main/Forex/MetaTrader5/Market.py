import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from Forex.MetaTrader5.Symbols import symbols


class MarketData:
    def __init__(self,
                 c_datetime_request,
                 c_symbols: list = symbols,
                 ):
        self.symbols = c_symbols
        self.datetime_request = c_datetime_request
        self.ticks_time = pd.DataFrame(columns=['time'])
        self.all_rates = self.get_rates()

    def get_rates(self):
        all_p = []

        for i, sym in enumerate(self.symbols):
            mt5_rates = mt5.copy_rates_range(sym, *self.datetime_request)

            df_rates = pd.DataFrame(mt5_rates)
            if i == 0:
                self.ticks_time = df_rates['time']

            df_rates = df_rates.rename(columns=lambda c: f'{sym},{c}')
            all_p.append(df_rates.iloc[:, 1:7])

        all_p = pd.concat(all_p, axis=1)
        all_p = all_p.fillna(method='ffill')
        all_p = all_p.fillna(method='bfill')

        return all_p

    @property
    def prices(self, price_type='close'):
        all_prices = []
        for sym in self.symbols:
            all_prices.append(self.all_rates[f'{sym},{price_type}'])
        all_prices = pd.concat(all_prices, axis=1)

        return all_prices.to_numpy(dtype=float)

    @property
    def percentage_return(self, price_type='close'):
        all_prices = []
        for sym in self.symbols:
            all_prices.append(self.all_rates[f'{sym},{price_type}'])
        all_prices = pd.concat(all_prices, axis=1)

        pct_prices = all_prices.pct_change().fillna(method='bfill')
        return pct_prices.to_numpy(dtype=float)

    @property
    def log_return(self, price_type='close'):
        all_prices = []
        for sym in self.symbols:
            all_prices.append(self.all_rates[f'{sym},{price_type}'])
        all_prices = pd.concat(all_prices, axis=1)

        log_p = np.log(all_prices)
        log_p = log_p.diff().fillna(method='bfill')
        return log_p.to_numpy(dtype=float)

    @property
    def num_symbols(self):
        return len(self.symbols)

    @property
    def volume(self):
        all_volume = []
        for sym in self.symbols:
            all_volume.append(self.all_rates[f'{sym},tick_volume'])
        all_volume = pd.concat(all_volume, axis=1)
        return all_volume.to_numpy(dtype=float)

    @property
    def percentage_volume(self):
        all_volume = []
        for sym in self.symbols:
            all_volume.append(self.all_rates[f'{sym},tick_volume'])
        all_volume = pd.concat(all_volume, axis=1)
        diff_vol = all_volume.pct_change().fillna(method='bfill')

        return diff_vol.to_numpy(dtype=float)

    @property
    def time(self):
        return pd.to_datetime(self.ticks_time, unit='s')

    @property
    def log_true_range(self):

        # self.all_log_high_low = self.all_log_high_low.fillna(method='bfill')
        # self.all_log_high_low = self.all_log_high_low.fillna(method='ffill')
        # return self.all_log_high_low.to_numpy(dtype=float)
        return []


class LiveTicks:
    def __init__(self,
                 c_datetime_request,
                 c_diff_datetime,
                 c_symbol: str,
                 ):
        self.datetime_request = c_datetime_request
        self.delta_datetime = c_diff_datetime
        self.symbol = c_symbol
        self.ticks = self.get_ticks()

    def get_ticks(self):
        start_datetime = self.datetime_request - self.delta_datetime
        end_datetime = self.datetime_request
        all_ticks = pd.DataFrame(mt5.copy_ticks_range(self.symbol, start_datetime, end_datetime, mt5.COPY_TICKS_ALL))

        class ticks_sides(object):
            bid = all_ticks['bid'].to_numpy(dtype=float)
            ask = all_ticks['ask'].to_numpy(dtype=float)

        return ticks_sides
