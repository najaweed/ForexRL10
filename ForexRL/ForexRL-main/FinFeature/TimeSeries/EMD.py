import pandas as pd
from PyEMD import EEMD
import numpy as np
from PyEMD import CEEMDAN
from statsmodels.tsa.stattools import adfuller
import antropy as ant


def imfs(time_series: np.ndarray, parallel=True):
    time_vector = np.linspace(0, 1, len(time_series))
    eemd = CEEMDAN(trials=100, parallel=parallel)
    all_imfs = eemd.ceemdan(time_series, time_vector)
    return all_imfs


def de_trending(time_series, imfs):
    trend = np.zeros_like(time_series)
    ad_test = adfuller(time_series, autolag="AIC")

    for k in range(1, len(imfs)):
        if ad_test[1] >= 0.05:
            trend += imfs[-k]
            ad_test = adfuller(time_series - trend, autolag="AIC")

        elif ad_test[1] < 0.05:
            return trend

    return trend


def de_fractal(time_series, imfs):
    fractal = np.zeros_like(time_series)
    fractal_test = ant.higuchi_fd(time_series)
    for k in range(len(imfs)):
        if fractal_test >= 1.03:
            fractal += imfs[k]
            fractal_test = ant.higuchi_fd(time_series - fractal)

        elif fractal_test < 1.03:
            return time_series - fractal

    return time_series - fractal


def de_noise(time_series, imfs, number_reduction_mode: int = 4):
    noise = np.zeros_like(time_series)
    for k in range(number_reduction_mode):
        noise += imfs[k]

    return time_series - noise


from FinFeature.OfflineMarket import OfflineMarket
import time

df = pd.read_csv(f'/home/z/Desktop/backups/memory_backup/DB/rates.csv')
market = OfflineMarket(df)
# print(market.prices('high')[:,0])
# z = np.linspace(-30*np.pi, 30*np.pi, 2001)
# x = np.linspace(-10*np.pi, 10*np.pi, 2001)
# y = np.linspace(-2*np.pi, 2*np.pi, 2001)
# ts = np.sin(y+np.sin(x + np.sin(z)))

import matplotlib.pyplot as plt

from scipy.signal import hilbert, chirp
import random


def Trans_Hilbert(time_series):
    signal1 = np.concatenate((time_series[::-1], time_series))

    analytic_signal_i = hilbert(signal1)[len(time_series):]  #

    amplitude_envelope_i = np.abs(analytic_signal_i)
    instantaneous_phase_i = np.unwrap(np.angle(analytic_signal_i))
    instantaneous_frq = np.diff(instantaneous_phase_i)

    return amplitude_envelope_i, instantaneous_phase_i, instantaneous_frq

ts = market.volume[:(2*5 * 24 * 12), 1]

x = np.linspace(0, ts.shape[0], ts.shape[0])


def multi_wave(p_omega_waves, time_steps):
    phases = [2 * np.pi * random.random() for _ in p_omega_waves]
    p_omega_waves = [random.gauss(w, 0.001) for w in p_omega_waves]
    decays = [random.gauss(0, 0.0001) for _ in p_omega_waves]

    w_lead = np.zeros_like(time_steps)
    for i in range(len(p_omega_waves)):
        w_lead = np.exp(decays[i] * time_steps) * np.sin(w_lead + p_omega_waves[i] * time_steps + phases[i])
    return w_lead


omegas = [2 * np.pi * (1 / (12 * 6)),2 * np.pi * (1 / (12 * 12)), 2 * np.pi * (1 / (12 * 24))]
ts1 = multi_wave(omegas, x)

modes = imfs(ts)
signal = np.zeros_like(ts)
for i, mode in enumerate(list(reversed(modes))):  # modes:  # list(reversed(modes)):
    signal += mode
    amplitude_envelope, instantaneous_phase, instantaneous_frq = Trans_Hilbert(signal)

    fig, (ax0, ax1 , ax2 ) = plt.subplots(nrows=3)

    ax0.plot(signal, label='signal')

    ax0.plot(amplitude_envelope, label='envelope')

    ax0.set_xlabel("time in seconds")

    ax0.legend()

    ax1.plot(instantaneous_phase)
    ax2.plot(np.sin(instantaneous_phase))



    fig.tight_layout()
    plt.show()
