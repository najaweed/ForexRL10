import pandas as pd
from PyEMD import EEMD
import numpy as np
from PyEMD import CEEMDAN
from statsmodels.tsa.stattools import adfuller
import antropy as ant


def imfs(time_series: np.ndarray):
    time_vector = np.linspace(0, 1, len(time_series))
    eemd = CEEMDAN(trials=3)
    all_imfs = eemd.ceemdan(time_series, time_vector)
    return all_imfs


def de_trending(time_series, imfs):
    trend = np.zeros_like(time_series)
    ad_test = adfuller(time_series, autolag="AIC")

    for k in range(1,len(imfs)):
        if ad_test[1] >= 0.05:
            trend +=imfs[-k]
            ad_test = adfuller(time_series - trend, autolag="AIC")

        elif ad_test[1] < 0.05:
            return trend

    return trend





def de_fractal(time_series, imfs):
    fractal = np.zeros_like(time_series)
    fractal_test = ant.higuchi_fd(time_series)
    for k in range(len(imfs)):
        if fractal_test >=1.03:
            fractal +=imfs[k]
            fractal_test = ant.higuchi_fd(time_series - fractal)

        elif fractal_test < 1.03:
            return time_series - fractal

    return time_series - fractal



def de_noise(time_series, imfs , number_reduction_mode:int = 4):
    noise = np.zeros_like(time_series)
    for k in range(number_reduction_mode):
        noise +=imfs[k]

    return time_series - noise