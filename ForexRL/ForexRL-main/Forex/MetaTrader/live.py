from datetime import datetime
import MetaTrader5 as mt5
import numpy as np
import pytz
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import time

if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

currency = ['USD', 'EUR', 'GBP', 'AUD', 'NZD', 'JPY', 'CHF', 'CAD']#, 'HKD', 'XAU', 'XAG', 'BTC', 'ETH']

pairs = ['USDJPY', 'USDCHF', 'USDCAD', 'USDHKD', 'EURUSD', 'EURGBP', 'EURAUD', 'EURNZD', 'EURJPY', 'EURCHF', 'EURCAD',
         'EURHKD', 'GBPUSD', 'GBPAUD', 'GBPNZD', 'GBPJPY', 'GBPCHF', 'GBPCAD', 'AUDUSD', 'AUDNZD', 'AUDJPY', 'AUDCHF',
         'AUDCAD', 'NZDUSD', 'NZDJPY', 'NZDCHF', 'NZDCAD', 'CHFJPY', 'CADJPY', 'CADCHF', 'XAUUSD', 'XAGUSD', 'BTCUSD',
         'BTCEUR', 'BTCJPY', 'ETHUSD', 'ETHBTC']
valid_pairs = pairs
#print(valid_pairs)

# TODO get prcie by datetime format and set window_size and time_frame as params
def symbol_price(symbol):
    # rates from mt5
    # mt5_rates = mt5.copy_rates_range(symbol, time_frame, start_datetime, finish_datetime)
    mt5_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 60 * 6)
    # convert to data frame
    #print("get price", f'{symbol}')
    df_rates = pd.DataFrame(mt5_rates)
    df_rates.fillna(method='ffill')
    df_rates.fillna(method='bfill')
    # ohlc price mean -- avg price
    df_rates = df_rates.iloc[:, 1:5]
    df_avg = df_rates.mean(axis=1)
    # print(df_avg)
    price_cols = df_rates.columns.values
    # print(price_cols)
    v = df_avg.to_numpy()
    df = pd.DataFrame()
    df[f'{symbol}'] = df_avg
    for i in range(len(price_cols)):
        df[f'{symbol},{price_cols[i]}'] = df_rates.iloc[:, i]
    # print(df)
    return df  # v#(v - v.min()) / (v.max() - v.min())  # np.diff(np.log(df_avg.to_numpy()))

def df_price():
    all_p = pd.DataFrame()
    all_p = [symbol_price(pairs[i]) for i in range(len(pairs))]
    all_p = pd.concat(all_p, axis=1)

    #print(all_p.isnull().values.any())
    all_p = all_p.fillna(method='bfill')
    all_p = all_p.fillna(method='ffill')
    return all_p
# print(all_p.isnull().values.any())
# print(all_p)

# all_p.to_csv("D:\\all_raw_price3.csv", mode="w")

#     all_p[f'{pairs[i]}'] =symbol_price(pairs[i])

# print(all_p.isnull().values.any())
# all_p=all_p.fillna(method='bfill')
# all_p=all_p.fillna(method='ffill')
# print(all_p.isnull().values.any())
# #all_p.to_csv("G:\My Drive\\all_raw_price2.csv", mode="w")
# print('sleep 30 sec')
# print(all_p['AUDNZD'])