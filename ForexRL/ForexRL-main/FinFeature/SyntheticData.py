import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

df_rates = pd.read_csv('DB\\rates.csv')
df_smooth_rates = pd.read_csv('DB\\smooths.csv')
print(df_rates - df_smooth_rates)
df_delta_noise = df_rates - df_smooth_rates
df_size = df_smooth_rates.shape[0]
def synthetic_rates_to_csv(number_sample:int=10):

    for i in range(number_sample):
        df_synthetic = pd.DataFrame(columns=df_rates.columns)
        for col in df_rates:
            print(col)
            mean, std = st.norm.fit(df_delta_noise[col])
            df_synthetic[col] = df_smooth_rates[col] + st.norm(mean, std / 2).rvs(size=df_size)
        df_synthetic.to_csv(f'DB\\sample_{i}.csv')

synthetic_rates_to_csv(20)