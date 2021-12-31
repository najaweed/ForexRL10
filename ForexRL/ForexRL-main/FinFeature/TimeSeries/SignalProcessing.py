import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from FinFeature.OfflineMarket import OfflineMarket
import time
import pandas as pd

df = pd.read_csv(f'/home/z/Desktop/backups/memory_backup/DB/rates.csv')
market = OfflineMarket(df)

x = market.volume[2000:(2000 + (5 * 24 * 12)), 27]
print('num sumple', len(x))
t = np.linspace(0, 1, len(x))

low_cut = 5 * 14 / (len(x) / 2)
high_cut = 5 * 20 / (len(x) / 2)
print('band freq', low_cut, high_cut)
#
sig = signal.butter(N=4, Wn=[low_cut, high_cut], btype='bandpass')

b = sig[0]
a = sig[1]

y1 = signal.filtfilt(b, a, x)

plt.plot(t, x, 'b')

plt.plot(t, y1, 'k')

plt.grid(True)

plt.show()

from scipy.fft import fft, fftfreq

# Number of sample points

N = len(x)

# sample spacing

T = 1.0 / len(x)

# t = np.linspace(0.0, N*T, N, endpoint=False)

y = y1

yf = fft(y)

xf = fftfreq(N, T)[:N // 2]

import matplotlib.pyplot as plt

plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))

plt.grid()

plt.show()
