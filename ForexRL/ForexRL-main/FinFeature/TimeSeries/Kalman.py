from filterpy.kalman import FixedLagSmoother
import numpy as np
import pandas as pd


def Kalman_smoother(price:np.ndarray , Q = 0.0000001):
    fls = FixedLagSmoother(dim_x=2, dim_z=1)

    fls.x = np.array([[price[0]],
                      [.0]])

    fls.F = np.array([[1., 1.],
                      [0., 1.]])

    fls.H = np.array([[1., 0.]])

    fls.P *= 200
    fls.R *= 5.
    fls.Q *= Q

    xhatsmooth, xhat = fls.smooth_batch(price, N=2)

    smooths1 = pd.DataFrame(xhatsmooth[:, 0])
    smooth_p = smooths1.to_numpy().flatten()

    return smooth_p.astype(dtype=np.float)
