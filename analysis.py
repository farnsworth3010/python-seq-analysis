"""Time series analysis module"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from data import t as t_orig, y as y_orig


def run_analysis():
    """Run time series analysis on the provided data."""
    t = np.array(t_orig)
    y = np.array(y_orig)

    # --- Linear trend ---
    coeffs = np.polyfit(t, y, 1)
    trend = np.polyval(coeffs, t)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # First plot: original data
    axs[0].plot(t, y, "o-", label="Actual data")
    axs[0].set_xlabel("Month")
    axs[0].set_ylabel("Revenue, mln RUB")
    axs[0].set_title("Revenue time series")
    axs[0].legend()

    # Second plot: trends
    axs[1].plot(t, y, "o", label="Actual data")
    axs[1].plot(t, trend, "-", label="Linear trend")

    # --- Seasonal trend (sinusoidal approximation) ---
    def seasonal_func(x, a, b, w, phi, c):
        return a + b * np.sin(w * x + phi) + c * x

    # Initial guess for parameters
    guess = [np.mean(y), (max(y) - min(y)) / 2, 2 * np.pi / len(t), 0, 0]
    params, _ = curve_fit(seasonal_func, t, y, p0=guess)
    seasonal_trend = seasonal_func(t, *params)

    t_future = 11
    y_pred_linear = np.polyval(coeffs, t_future)  
    y_pred_seasonal = seasonal_func(t_future, *params)

    t = np.append(t, t_future)
    y = np.append(y, y_pred_seasonal)
    params, _ = curve_fit(seasonal_func, t, y, p0=guess)
    seasonal_trend = seasonal_func(t, *params)

    axs[1].plot(t, seasonal_trend, "--", label="Seasonal trend (sine)")

    axs[1].set_xlabel("Month")
    axs[1].set_ylabel("Revenue, mln RUB")
    axs[1].set_title("Trends: linear and seasonal")
    axs[1].legend()

    plt.tight_layout()


    print(f"Revenue forecast for month 11 (linear trend): {y_pred_linear:.2f} mln RUB")
    print(
        f"Revenue forecast for month 11 (seasonal trend): {y_pred_seasonal:.2f} mln RUB"
    )

    plt.show()
