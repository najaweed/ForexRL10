import pandas as pd
import numpy as np


class OfflineMarketTrend:
    def __init__(self,
                 c_data_frame,
                 c_symbols,
                 ):
        self.all_rates = c_data_frame
        self.symbols = c_symbols

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

    # @property
    # def volume(self):
    #     all_volume = []
    #     for sym in self.symbols:
    #         all_volume.append(self.all_rates[f'{sym},tick_volume'])
    #     all_volume = pd.concat(all_volume, axis=1)
    #     return all_volume.to_numpy(dtype=float)
    #
    # @property
    # def percentage_volume(self):
    #     all_volume = []
    #     for sym in self.symbols:
    #         all_volume.append(self.all_rates[f'{sym},tick_volume'])
    #     all_volume = pd.concat(all_volume, axis=1)
    #     diff_vol = all_volume.pct_change().fillna(method='bfill')
    #
    #     return diff_vol.to_numpy(dtype=float)
    #
    # @property
    # def log_true_range(self):
    #
    #     # self.all_log_high_low = self.all_log_high_low.fillna(method='bfill')
    #     # self.all_log_high_low = self.all_log_high_low.fillna(method='ffill')
    #     # return self.all_log_high_low.to_numpy(dtype=float)
    #     return []
    #
    pass
