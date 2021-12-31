import pandas as pd
import numpy as np

currencies = ['USD', 'EUR', 'GBP', 'AUD', 'NZD', 'JPY', 'CHF', 'CAD']  # , 'HKD', 'XAU', 'XAG', 'BTC', 'ETH']


class OfflineMarket:

    def __init__(self,
                 c_data_frame,
                 ):
        self.all_rates = c_data_frame
        self.symbols = self.ordered_symbols()
        self.decimal_points = self.price_decimal()

    def get_symbols(self):
        symbols = []
        for col in self.all_rates.columns:
            symbols.append(col[:8])
        x_symbols = []
        for sym in symbols:
            if sym not in x_symbols:
                x_symbols.append(sym)

        return x_symbols

    def ordered_symbols(self):
        syms = self.get_symbols()

        ordered_symbols = [[] for _ in currencies]
        for sym in syms:
            ordered_symbols[currencies.index(sym[3:6])].append(sym)
        flat_list = [item for sublist in ordered_symbols for item in sublist]
        return flat_list

    def ohlcv(self, symbol):
        p_time = pd.DataFrame()
        p_time['Date'] = pd.to_datetime(self.all_rates.iloc[:, 0], unit='s')

        all_prices = [p_time]
        for r_type in ['open', 'high', 'low', 'close', 'tick_volume']:
            p_price = pd.DataFrame()
            p_type = 'volume' if r_type == 'tick_volume' else r_type
            p_price[f'{p_type}'] = self.all_rates[f'{symbol},{r_type}']
            all_prices.append(p_price)

        all_prices = pd.concat(all_prices, axis=1)
        all_prices = all_prices.set_index('Date')
        all_prices = all_prices[~all_prices.index.duplicated(keep='first')]
        all_prices = all_prices.reindex(pd.date_range(all_prices.index[0], all_prices.index[-1], freq='min'))
        all_prices = all_prices.fillna(method='bfill')
        all_prices = all_prices.fillna(method='ffill')

        return all_prices

    def price_decimal(self):
        decimal_points = np.zeros(self.num_symbols, dtype=int)
        for i in range(self.num_symbols):
            f = str(self.prices('close')[10, i])
            dec = f[::-1].find('.')
            dec = 3 if dec == 2 else dec
            dec = 5 if dec == 4 else dec
            decimal_points[i] = 10 ** dec
        return decimal_points

    def prices(self, price_type='close'):
        all_prices = []
        for sym in self.symbols:
            all_prices.append(self.all_rates[f'{sym},{price_type}'])
        all_prices = pd.concat(all_prices, axis=1)

        return all_prices.to_numpy(dtype=float)

    def percentage_return(self, price_type='close'):
        all_prices = []
        for sym in self.symbols:
            all_prices.append(self.all_rates[f'{sym},{price_type}'])
        all_prices = pd.concat(all_prices, axis=1)

        pct_prices = all_prices.pct_change().fillna(method='bfill')
        return pct_prices.to_numpy(dtype=float)

    def price_point_diff(self, price_type='close'):
        all_prices = []
        for sym in self.symbols:
            all_prices.append(self.all_rates[f'{sym},{price_type}'])
        all_prices = pd.concat(all_prices, axis=1)
        prices = all_prices
        diff_prices = prices.diff().fillna(method='bfill')
        diff_point = diff_prices.to_numpy(dtype=float)
        return self.decimal_points * diff_point

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
    def log_true_range(self):

        # self.all_log_high_low = self.all_log_high_low.fillna(method='bfill')
        # self.all_log_high_low = self.all_log_high_low.fillna(method='ffill')
        # return self.all_log_high_low.to_numpy(dtype=float)
        return []

    def multi_time_rates(self, price_type='high', time_shape=(8, 4, 5, 24, 12), ):
        """ out-size : time_shape x ( num_symbols)
        """
        rates = self.log_return(price_type=price_type)
        # TODO check input shape and adjust with rate shape
        # if rates.shape[0] < np.prod(list(time_shape)):
        #     print(np.prod(list(time_shape[-4:]))/rates.shape[0])
        #
        tensor_rates = np.zeros((time_shape + (self.num_symbols,)))

        for month in range(time_shape[0]):
            for week in range(time_shape[1]):
                for day in range(time_shape[2]):
                    for hour in range(time_shape[3]):
                        for min5 in range(time_shape[4]):
                            index_time = 0
                            index_time += min5
                            index_time += hour * np.prod(list(time_shape[-1:]))
                            index_time += day * np.prod(list(time_shape[-2:]))
                            index_time += week * np.prod(list(time_shape[-3:]))
                            index_time += month * np.prod(list(time_shape[-4:]))

                            if index_time < rates.shape[0]:
                                tensor_rates[month, week, day, hour, min5, :] = rates[index_time, :]

        return tensor_rates

    def tensor_rates_for_cnn(self, time_shape=(8, 4, 5, 24, 12), ):
        """
        out.shape

        """
        features = ['low', 'close', 'high'] #, 'tick_volume']

        tensor_rates = np.zeros((time_shape[0], len(features), *list(time_shape)[1:], self.num_symbols))
        # print(tensor_rates.shape)

        for i, feature in enumerate(features):
            tensor_rates[:, i, ...] = self.multi_time_rates(feature , time_shape=time_shape)
        return np.squeeze(tensor_rates)


#
# import matplotlib.pyplot as plt
#
# df = pd.read_csv(f'/home/z/Desktop/backups/memory_backup/DB/rates.csv')
# market = OfflineMarket(df)
#
#
# plt.plot(market.prices()[:,1])
# plt.show()

# rates = rates[..., 0]
# #plt.plot(rates.flatten())
# plt.plot(rates.flatten() - np.ones(rates.flatten().shape[0]) * np.mean(rates.flatten()))
# smooth_rates = np.copy(rates)
# for i in range(1, 5):
#     mean_dim = np.mean(smooth_rates, keepdims=True, axis =-i)
#     smooth_rates[..., :] = np.ones(rates.shape[-i:]) * mean_dim
#
#     plt.plot(smooth_rates.flatten() - np.ones(rates.flatten().shape[0]) * np.mean(rates.flatten()))
# #plt.plot(np.ones(rates.flatten().shape[0]) * np.mean(rates.flatten()))
# plt.show()
