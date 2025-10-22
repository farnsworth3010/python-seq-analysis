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
    # Formula: y = a * t + b
    # Fits a straight line to the data (models steady growth or decline)
    coeffs = np.polyfit(t, y, 1)
    trend = np.polyval(coeffs, t)

    # --- Visualization ---
    # Open two plots side by side
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
    # Many real-world time series have seasonality (repeating patterns),
    # for example, higher sales in winter for winter sports equipment.
    # A sine wave models this periodic behavior.
    #
    # Formula: y = a + b * sin(w * t + phi) + c * t
    # Where:
    #   a    - baseline (average level), e.g. a = 50 (if average revenue is 50 mln RUB)
    #   b    - amplitude (strength of seasonal effect), e.g. b = 40 (if seasonal swing is ±40)
    #   w    - angular frequency (how often the cycle repeats), e.g. w = 2*pi/10 ≈ 0.63 (for 10-month cycle)
    #   phi  - phase shift (where the peak occurs), e.g. phi = -w*2 (if peak at t=2)
    #   c    - linear trend component (overall increase/decrease), e.g. c = 2 (if revenue grows by 2 mln RUB per month)
    #
    # Example: For t = 5, a = 50, b = 40, w = 0.63, phi = -1.26, c = 2:
    #   y = 50 + 40 * sin(0.63 * 5 - 1.26) + 2 * 5
    def seasonal_func(x, a, b, w, phi, c):
        return a + b * np.sin(w * x + phi) + c * x

    # Initial guess for parameters: [a, b, w, phi, c]
    guess = [np.mean(y), (max(y) - min(y)) / 2, 2 * np.pi / len(t), 0, 0]
    # Fit the seasonal model to the data
    params, _ = curve_fit(seasonal_func, t, y, p0=guess)
    seasonal_trend = seasonal_func(t, *params)

    t_future = 11
    # Use the fitted linear model to predict revenue for month 11
    y_pred_linear = np.polyval(coeffs, t_future)  # Linear trend forecast
    y_pred_seasonal = seasonal_func(t_future, *params)  # Seasonal trend forecast

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

    # --- Forecast for next month ---
    print(f"Revenue forecast for month 11 (linear trend): {y_pred_linear:.2f} mln RUB")
    print(
        f"Revenue forecast for month 11 (seasonal trend): {y_pred_seasonal:.2f} mln RUB"
    )

    plt.show()
