import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from Forex.MetaTrader5.Market import MarketData
from Forex.MetaTrader5.Time import MarketTime
from Forex.MetaTrader5.Symbols import Symbols, currencies
from FinFeature.TimeSeries.EMD import E_Modes

mrk = MarketData(MarketTime().time_range, Symbols(currencies).selected_symbols)


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


# print(hilbert_transform(mrk.log_return[:400, 0])[2])

fin_feature = mrk.volume[:6400, 0]
de_log = E_Modes(fin_feature)
de_frq = hilbert_transform(de_log).freq
frq = hilbert_transform(fin_feature).freq

x = np.linspace(0, len(fin_feature), len(fin_feature))
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(fin_feature)
axs[0].plot(de_log)
axs[1].scatter(x, frq, s=1.5)
axs[1].scatter(x, de_frq, s=1.5)

plt.show()
