import time

import numpy as np
from scipy import signal
import pandas as pd
from scipy.signal import hilbert, chirp
from scipy.signal import argrelextrema

from filterpy.kalman import FixedLagSmoother
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from filterpy.kalman import FixedLagSmoother


def ewma(data, window):
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha

    scale = 1 / alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale ** r
    offset = data[0] * alpha_rev ** (r + 1)
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def low_pass_filter(xs, alpha=0.8):
    # low_cut = band[0] / (len(xs) / 2)

    xs = np.concatenate((xs, xs[::-1]))
    sig = signal.butter(N=4, Wn=alpha, btype='lowpass')
    b = sig[0]
    a = sig[1]
    y1 = signal.filtfilt(b, a, xs)

    xs = y1[:int(len(xs) / 2)]
    return xs


class SignalSeasonal:
    def __init__(self,
                 q_modes=(0.00001, 0.01)
                 # config,
                 ):
        # self.window_size = config['window']
        # self.freq_modes = config['freq_modes']
        self.q_modes = q_modes

    @staticmethod
    def z_score(xs, mean):
        var = sum(pow(xx - mean, 2) for xx in xs) / len(xs)  # variance
        std = np.sqrt(var)  # standard deviation
        return (xs - mean) / std

    @staticmethod
    def low_pass_filter(self, xs, band=80):
        # low_cut = band[0] / (len(xs) / 2)
        high_cut = band / (len(xs) / 2)
        xs = np.concatenate((xs, xs[::-1]))
        sig = signal.butter(N=4, Wn=high_cut, btype='lowpass')
        b = sig[0]
        a = sig[1]
        y1 = signal.filtfilt(b, a, xs)

        xs = y1[:int(len(xs) / 2)]
        return xs

    def high_pass_filter(self, xs, band=24):
        low_cut = band / (len(xs) / 2)
        # high_cut = band / (len(xs) / 2)
        xs = np.concatenate((xs, xs[::-1]))
        sig = signal.butter(N=4, Wn=low_cut, btype='highpass')
        b = sig[0]
        a = sig[1]
        y1 = signal.filtfilt(b, a, xs)

        xs = y1[:int(len(xs) / 2)]
        return xs

    def band_pass_filter(self, xs, band=(12, 80)):
        low_cut = band[0] / (len(xs) / 2)
        high_cut = band[1] / (len(xs) / 2)
        xs = np.concatenate((xs, xs[::-1]))
        sig = signal.butter(N=4, Wn=(low_cut, high_cut), btype='bandpass')
        b = sig[0]
        a = sig[1]
        y1 = signal.filtfilt(b, a, xs)

        xs = y1[:int(len(xs) / 2)]
        return xs

    def kalman_smooth(self, x: np.ndarray, q_var=0.00001):
        fk = FixedLagSmoother(dim_x=2, dim_z=1)
        x0 = x[0]
        fk.x = np.array([float(x0), 0.])  # state (x and dx)

        fk.F = np.array([[1., 1.],
                         [0., 1.]])  # state transition matrix

        fk.H = np.array([[1., 0.]])  # Measurement function
        fk.P *= 1  # covariance matrix
        fk.R = 1  # state uncertainty
        fk.Q = Q_discrete_white_noise(dim=2, dt=1., var=q_var)  # process uncertainty

        # create noisy data

        # filter data with Kalman filter, than run smoother on it
        mu, cov = fk.smooth_batch(x, N=2)
        # M, P, C, _ = fk.rts_smoother(mu, cov)
        # print(P.shape)
        return mu[:, 0]

    def trade_modes(self, prices):
        modes = []

        mode_low = self.kalman_smooth(prices, self.q_modes[0])  # self.low_pass_filter(prices, 2)#
        kalman_smooth_low = self.kalman_smooth(prices, self.q_modes[1])

        modes.append(mode_low)
        modes.append(kalman_smooth_low)

        return modes

    @staticmethod
    def hilbert_transform(x_i):
        signal1 = np.concatenate((x_i[::-1], x_i))

        analytic_signal = hilbert(signal1)[len(x_i):]
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        instantaneous_frq = np.diff(np.unwrap(instantaneous_phase))
        instantaneous_frq = np.insert(instantaneous_frq, 0, instantaneous_frq[0])

        class output:
            amp = amplitude_envelope
            phase = instantaneous_phase
            freq = instantaneous_frq

        return output


# from FinFeature.OfflineMarket import OfflineMarket
# import matplotlib.pyplot as plt
#
# df = pd.read_csv(f'./ratesM1.csv')
# market = OfflineMarket(df)
#
# ss_config = {
#     "window": (1 * 6 * 60),
#     "freq_modes": [4, 15, 60, 240, int((1 * 6 * 60) / 2 - 1)],
# }
#
# for frq in ss_config['freq_modes']:
#     print(int(ss_config['window'] / frq))
# # fig, axs = plt.subplots(4)
# symbol = 'EURUSD.c'
# win = ss_config['window']
# ss = SignalSeasonal((0.00005, 0.01))
#
# fig, ax = plt.subplots(2)
# #
# df = market.ohlcv(symbol)
#
# for t in range(ss_config['window'] + 200, ss_config['window'] + 600, 1):
#
#     ts = df.iloc[(t - ss_config['window']):t, 0].to_numpy()
#     # #
#     print()
#     modes = ss.trade_modes(ts)
#     ax[0].plot(ts)
#     ax[0].plot(modes[0])
#     ax[0].plot(modes[1])
#     # ax[0].plot( modes[1])
#
#     ax[1].plot(ts - modes[0])
#     ax[1].plot(ss.z_score(ts, modes[1]))
#     ax[1].plot(ss.z_score(ts, modes[1]))
#     ax[1].plot(ss.z_score(modes[1], modes[0]))
#     z_score = ss.z_score(modes[1], modes[0])
#     plt.show()
#     breakpoint()
#     for i, z in enumerate(z_score):
#
#         if z > 0.2:
#             if z_score[i] < z_score[i - 1]:  # < z_score[i-2]:
#                 # ax[1].plot(i, z_score[i], 'rs')
#                 plt.plot(i, ts[i], 'rs')
#         elif z < -0.2:
#             if z_score[i] > z_score[i - 1]:  # > z_score[i-2]:
#                 # ax[1].plot(i, z_score[i], 'rs')
#                 plt.plot(i, ts[i], 'rs')
#     # for i in range(1, len(modes)):
#     #     ax[1].plot(modes[i]-modes[i-1])
#     #     # ax[1].ylim([-0.0003,0.0003])
#     plt.show(block=False)
#     plt.pause(1.5)
#     plt.close()
#
# t_index = df.iloc[(t - ss_config['window']):t, 4].index
# low_pass = ss.low_pass_filter(ts, 2)
# xc = ts - low_pass
# xc *= (1e5)
# plt.plot(xc)
# plt.plot(ss.smooth(xc, 0.001))
#
# plt.plot(xc - ss.smooth(xc, 0.001))
#
# plt.show()

# print(modes)
# # plt.figure(2)
# plt.plot(t_index, ts-ts[0])
# plt.show()
# axs[0].plot(t_index, modes[0])
# axs[0].plot(t_index, modes[0])

# for i in range(len(modes)):
#     # plt.figure(1)
#     # plt.set_title(f'{symbol}')
#     # axs[1].plot(t_index, hilbert_transform(modes[i]).amp)
#     # axs[1].axhline(np.mean(hilbert_transform(modes[i]).amp))
#     axs[1].plot(t_index, modes[i])
#
#
#
# for i in range(1, len(modes)):
#     #axs[1].plot(t_index, (modes[i] ))
#     #axs[1].plot(t_index, numpy_ewma_vectorized(hilbert_transform(modes[i] - modes[i - 1]).amp, 20))
#     #if i !=len(modes)-1:
#     axs[2].plot(t_index,ss.z_score(modes[i],modes[i-1]))
#     #axs[2].plot(t_index, abs(hilbert_transform(ss.z_score(modes[i],modes[i-1])).freq))
#     #axs[2].plot([t_index[100],t_index[200]],[200,400],'rs')
#     #axs[2].plot(t_index, ewma(abs(hilbert_transform(ss.z_score(modes[i],modes[i-1])).freq),60))
#     #axs[2].axhline(np.mean(ewma(abs(hilbert_transform(modes[i] ).freq),60)))
#
# for i in range(1, len(modes)):
#     axs[3].plot(t_index, modes[i] - modes[i - 1])
#
# plt.show()

# for j, sym in enumerate(['AUDNZD.c']):
#
#     ss = SignalSeasonal(ss_config)
#     symbol = sym
#     win = ss_config['window']
#
#     # fig, axs = plt.subplots(5)
#     # #
#     df = market.ohlcv(symbol)
#     ts = df.iloc[3 * ss_config['window']:4 * ss_config['window'], 1].to_numpy()
#     # #
#     t_index = df.iloc[3 * ss_config['window']:4 * ss_config['window'], 4].index
#     # axs[0].plot(t_index, ts)
#     # axs[0].plot(t_index, ss.low_pass_filter(ts, 4))
#     # axs[0].plot(t_index, ss.low_pass_filter(ts, 15))
#     # axs[0].plot(t_index, ss.low_pass_filter(ts, 48))
#     # axs[0].plot(t_index, ss.low_pass_filter(ts, 80))
#     #
#     # axs[1].plot(t_index, ss.high_pass_filter(ts, 4))
#     # axs[1].plot(t_index, ss.band_pass_filter(ts, (4, 15)))
#     # axs[1].plot(t_index, ss.band_pass_filter(ts, (15, 48)))
#     # axs[1].plot(t_index, ss.band_pass_filter(ts, (48, 80)))
#     #
#     # axs[2].plot(t_index, ss.high_pass_filter(ts, 48), '--')
#     # axs[2].plot(t_index, ss.band_pass_filter(ts, (48, 80)), 'r')
#     #
#     # from scipy.signal import argrelextrema
#     #
#     # #
#     # # # x  = np.exp(np.linspace(0 , 0.9,10000))*np.sin(np.linspace(-20*np.pi , 20*np.pi,10000))
#     # # # plt.plot(x)
#     # # # fft_x = np.fft.fft(ss.band_pass_filter(ts, (24, 48)))
#     # # #
#     # #
#     # #
#     # #
#     # #
#     # # # # plt.plot(ss.high_pass_filter(ts, 48))
#     # # # print(np.argmax(np.abs(fft_x)[:int(len(fft_x) / 2)]))
#     # # #
#     # # # plt.plot(np.abs(fft_x)[:int(len(fft_x) / 2)])
#     # x = ss.band_pass_filter(ts, (48, 80))  # np.abs(fft_x)[:int(len(fft_x) / 2)]
#     # axs[2].plot(t_index, hilbert_transform(x).amp, 'g')
#     # axs[2].axhline(np.mean(hilbert_transform(x).amp))
#     # print(np.mean(hilbert_transform(x).amp))
#     # m = argrelextrema(x, np.greater, order=10)  # array of indexes of the locals maxima
#     # n = argrelextrema(x, np.less, order=10)  # array of indexes of the locals maxima
#     #
#     # y = [x[m] for i in m]
#     # y1 = [x[n] for j in n]
#     #
#     # axs[3].plot(m, y, 'rs')
#     # axs[3].plot(n, y1, 'rs')
#     #
#     # axs[3].plot(x)
#     #
#     # axs[3].axhline(np.mean(y))
#     # axs[3].axhline(np.mean(y1))
#     #
#     # ts = df.iloc[3 *ss_config['window']:4 * ss_config['window'], 4].to_numpy()
#     # axs[4].plot(t_index,ts)
#     #
#     # print(np.mean(y))
#     # print(np.mean(y1))
#     modes = ss.high_pass_modes(ts)
#     #axs[j].set_title(f'{symbol}')
#     #axs[j].plot(t_index,ts)
#     for i, mode in enumerate(modes[:3]):
#         #plt.figure(j)
#         axs[0].set_title(f'{symbol}')
#         axs[0].plot(t_index, mode)
#
# for j, sym in enumerate([ 'AUDCHF.c', 'AUDUSD.c',  ]):
#
#     ss = SignalSeasonal(ss_config)
#     symbol = sym
#     win = ss_config['window']
#
#     # fig, axs = plt.subplots(5)
#     # #
#     df = market.ohlcv(symbol)
#     ts = df.iloc[3 * ss_config['window']:4 * ss_config['window'], 1].to_numpy()
#     # #
#     t_index = df.iloc[3 * ss_config['window']:4 * ss_config['window'], 4].index
#     # axs[0].plot(t_index, ts)
#     # axs[0].plot(t_index, ss.low_pass_filter(ts, 4))
#     # axs[0].plot(t_index, ss.low_pass_filter(ts, 15))
#     # axs[0].plot(t_index, ss.low_pass_filter(ts, 48))
#     # axs[0].plot(t_index, ss.low_pass_filter(ts, 80))
#     #
#     # axs[1].plot(t_index, ss.high_pass_filter(ts, 4))
#     # axs[1].plot(t_index, ss.band_pass_filter(ts, (4, 15)))
#     # axs[1].plot(t_index, ss.band_pass_filter(ts, (15, 48)))
#     # axs[1].plot(t_index, ss.band_pass_filter(ts, (48, 80)))
#     #
#     # axs[2].plot(t_index, ss.high_pass_filter(ts, 48), '--')
#     # axs[2].plot(t_index, ss.band_pass_filter(ts, (48, 80)), 'r')
#     #
#     # from scipy.signal import argrelextrema
#     #
#     # #
#     # # # x  = np.exp(np.linspace(0 , 0.9,10000))*np.sin(np.linspace(-20*np.pi , 20*np.pi,10000))
#     # # # plt.plot(x)
#     # # # fft_x = np.fft.fft(ss.band_pass_filter(ts, (24, 48)))
#     # # #
#     # #
#     # #
#     # #
#     # #
#     # # # # plt.plot(ss.high_pass_filter(ts, 48))
#     # # # print(np.argmax(np.abs(fft_x)[:int(len(fft_x) / 2)]))
#     # # #
#     # # # plt.plot(np.abs(fft_x)[:int(len(fft_x) / 2)])
#     # x = ss.band_pass_filter(ts, (48, 80))  # np.abs(fft_x)[:int(len(fft_x) / 2)]
#     # axs[2].plot(t_index, hilbert_transform(x).amp, 'g')
#     # axs[2].axhline(np.mean(hilbert_transform(x).amp))
#     # print(np.mean(hilbert_transform(x).amp))
#     # m = argrelextrema(x, np.greater, order=10)  # array of indexes of the locals maxima
#     # n = argrelextrema(x, np.less, order=10)  # array of indexes of the locals maxima
#     #
#     # y = [x[m] for i in m]
#     # y1 = [x[n] for j in n]
#     #
#     # axs[3].plot(m, y, 'rs')
#     # axs[3].plot(n, y1, 'rs')
#     #
#     # axs[3].plot(x)
#     #
#     # axs[3].axhline(np.mean(y))
#     # axs[3].axhline(np.mean(y1))
#     #
#     # ts = df.iloc[3 *ss_config['window']:4 * ss_config['window'], 4].to_numpy()
#     # axs[4].plot(t_index,ts)
#     #
#     # print(np.mean(y))
#     # print(np.mean(y1))
#     modes = ss.high_pass_modes(ts)
#     #axs[j].set_title(f'{symbol}')
#     #axs[j].plot(t_index,ts)
#     for i, mode in enumerate(modes[:3]):
#         #plt.figure(j)
#         axs[1].set_title(f'{symbol}')
#         axs[1].plot(t_index, mode)
#
# for j, sym in enumerate([ 'NZDCHF.c', 'NZDUSD.c',  ]):
#
#     ss = SignalSeasonal(ss_config)
#     symbol = sym
#     win = ss_config['window']
#
#     # fig, axs = plt.subplots(5)
#     # #
#     df = market.ohlcv(symbol)
#     ts = df.iloc[3 * ss_config['window']:4 * ss_config['window'], 1].to_numpy()
#     # #
#     t_index = df.iloc[3 * ss_config['window']:4 * ss_config['window'], 4].index
#     # axs[0].plot(t_index, ts)
#     # axs[0].plot(t_index, ss.low_pass_filter(ts, 4))
#     # axs[0].plot(t_index, ss.low_pass_filter(ts, 15))
#     # axs[0].plot(t_index, ss.low_pass_filter(ts, 48))
#     # axs[0].plot(t_index, ss.low_pass_filter(ts, 80))
#     #
#     # axs[1].plot(t_index, ss.high_pass_filter(ts, 4))
#     # axs[1].plot(t_index, ss.band_pass_filter(ts, (4, 15)))
#     # axs[1].plot(t_index, ss.band_pass_filter(ts, (15, 48)))
#     # axs[1].plot(t_index, ss.band_pass_filter(ts, (48, 80)))
#     #
#     # axs[2].plot(t_index, ss.high_pass_filter(ts, 48), '--')
#     # axs[2].plot(t_index, ss.band_pass_filter(ts, (48, 80)), 'r')
#     #
#     # from scipy.signal import argrelextrema
#     #
#     # #
#     # # # x  = np.exp(np.linspace(0 , 0.9,10000))*np.sin(np.linspace(-20*np.pi , 20*np.pi,10000))
#     # # # plt.plot(x)
#     # # # fft_x = np.fft.fft(ss.band_pass_filter(ts, (24, 48)))
#     # # #
#     # #
#     # #
#     # #
#     # #
#     # # # # plt.plot(ss.high_pass_filter(ts, 48))
#     # # # print(np.argmax(np.abs(fft_x)[:int(len(fft_x) / 2)]))
#     # # #
#     # # # plt.plot(np.abs(fft_x)[:int(len(fft_x) / 2)])
#     # x = ss.band_pass_filter(ts, (48, 80))  # np.abs(fft_x)[:int(len(fft_x) / 2)]
#     # axs[2].plot(t_index, hilbert_transform(x).amp, 'g')
#     # axs[2].axhline(np.mean(hilbert_transform(x).amp))
#     # print(np.mean(hilbert_transform(x).amp))
#     # m = argrelextrema(x, np.greater, order=10)  # array of indexes of the locals maxima
#     # n = argrelextrema(x, np.less, order=10)  # array of indexes of the locals maxima
#     #
#     # y = [x[m] for i in m]
#     # y1 = [x[n] for j in n]
#     #
#     # axs[3].plot(m, y, 'rs')
#     # axs[3].plot(n, y1, 'rs')
#     #
#     # axs[3].plot(x)
#     #
#     # axs[3].axhline(np.mean(y))
#     # axs[3].axhline(np.mean(y1))
#     #
#     # ts = df.iloc[3 *ss_config['window']:4 * ss_config['window'], 4].to_numpy()
#     # axs[4].plot(t_index,ts)
#     #
#     # print(np.mean(y))
#     # print(np.mean(y1))
#     modes = ss.low_pass_modes(ts)
#     #axs[j].set_title(f'{symbol}')
#     #axs[j].plot(t_index,ts)
#     for i, mode in enumerate(modes[:3]):
#         #plt.figure(j)
#         axs[2].set_title(f'{symbol}')
#         axs[2].plot(t_index, mode)
#
# plt.show()

# modes = imfs(ss.high_pass_filter(ts , 24))
# x_mode =np.zeros_like(modes[0])
# for i in range(0,len(modes)):
#     x_mode +=modes[i]
#     plt.plot(x_mode)
#     plt.show()
# t=0
# for type_price in ['high', 'low']:
#
#     price = df.loc[(df.index[t]):(df.index[t + win]), type_price].to_numpy()
#     z_scores = ss.gen_mono_obs(price)
#     print(z_scores)
#     for j, z_s in enumerate(z_scores):
#         df.loc[df.index[t + win], f'z{j}_{type_price}'] = z_s
# print(df.loc[df.index[win],:])
#
# for t in range(0, 1000, 100):
#     df1 = df.iloc[(t + win):(t + 2 * win)]
#     fig, axs = plt.subplots(1, 2)
#     modesx = ss.seasonal_modes(df1['close'].to_numpy())
#     # ax = df1[['high', 'low']].plot()
#     axs[0].scatter(df1.index, df1['high'].to_numpy())
#     axs[0].scatter(df1.index, df1['low'].to_numpy())
#     # plt.plot(df1.index,modesx[3])
#     # axs[0].plot(df1.index,modesx[3] , c='tab:grey')
#
#     axs[0].plot(df1.index, modesx[2], c='r')
#     axs[0].plot(df1.index, modesx[1], c='g')
#     axs[1].plot()
#     df1[[z for z in df1.columns if z[0] == 'z']].plot(ax=axs[1])
#
#     plt.show(block=True)
#
#     plt.pause(10)
#     plt.close()

# for t in range(10000):
#     sig = np.zeros((2, 2))
#     print(t)
#     for i, ty in enumerate(['high', 'low']):
#         prices = market.prices(ty)[t:(t + t_range), :]
#
#         modes = ss.seasonal_modes(prices[:, 6])
#         sig[i, ...] = ss.modal_z_score(modes)
#     obs_to_csv[t, ...] = sig
#
# #np.save("obsEURUSD.npy", obs_to_csv, allow_pickle=True)
#
# print(obs_to_csv)
