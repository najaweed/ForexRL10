import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.stats as st
from scipy.signal import hilbert, chirp

# df_rates = pd.read_csv('DB\\rates.csv')
# df_smooth_rates = pd.read_csv('DB\\smooths.csv')
# print(df_rates - df_smooth_rates)
# df_delta_noise = df_rates - df_smooth_rates
# df_size = df_smooth_rates.shape[0]


# def synthetic_rates_to_csv(number_sample: int = 10):
#     for i in range(number_sample):
#         df_synthetic = pd.DataFrame(columns=df_rates.columns)
#         for col in df_rates:
#             print(col)
#             mean, std = st.norm.fit(df_delta_noise[col])
#             df_synthetic[col] = df_smooth_rates[col] + st.norm(mean, std / 2).rvs(size=df_size)
#         df_synthetic.to_csv(f'DB\\sample_{i}.csv')
#

# synthetic_rates_to_csv(20)


x = np.linspace(0, 1440, 1440)
print(len(x), x)


def Trans_Hilbert(time_serie):
    x_i = time_serie
    signal1 = x_i  # np.concatenate((x_i[::-1], x_i))

    analytic_signal_i = hilbert(signal1)  # [len(x_i):]
    amplitude_envelope_i = np.abs(analytic_signal_i)
    instantaneous_phase_i = np.angle(analytic_signal_i)
    instantaneous_frq = np.diff(np.unwrap(instantaneous_phase_i))

    return amplitude_envelope_i, instantaneous_phase_i, instantaneous_frq


w_1 = 2 * np.pi * (1 / (12 * 24))
w_2 = 2 * np.pi * (1 / (12 * 12))
w_3 = 2 * np.pi * (1 / (12 * 6))
ts = np.sin(w_1 * x + np.sin(w_2 * x + np.sin(w_3 * x)))


def multi_wave(p_omega_waves, time_steps):
    phases = [2 * np.pi * random.random() for _ in p_omega_waves]
    p_omega_waves = [random.gauss(w, 0.001) for w in p_omega_waves]
    decays = [random.gauss(0, 0.0005) for _ in p_omega_waves]

    w_lead = np.zeros_like(time_steps)
    for i in range(len(p_omega_waves)):
        w_lead = np.exp(decays[i]*time_steps)*np.sin(w_lead + p_omega_waves[i] * time_steps + phases[i])
    return w_lead

omegas = [2 * np.pi * (1 / (12 * 6)), 2 * np.pi * (1 / (12 * 12)), 2 * np.pi * (1 / (12 * 24))]
print('omegas' , omegas)
ts1 = multi_wave(omegas, x)

plt.plot(ts)
plt.plot(ts1)

plt.show()

# hil = Trans_Hilbert(ts)
# plt.plot(x, ts)
# plt.plot(x, hil[0])
#
# plt.xlabel('Angle [rad]')
#
# plt.ylabel('sin(x)')
#
# plt.axis('tight')
#
# plt.show()
# plt.plot(x[1:], hil[2])
#
# plt.show()
