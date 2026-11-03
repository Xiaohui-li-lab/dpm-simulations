#!/usr/bin/env python3
"""
dpm_benchmark.py
----------------
Benchmark DPM echo templates against LIGO/Virgo/KAGRA design sensitivity
using matched filtering in Gaussian noise.
"""

import numpy as np
from scipy.interpolate import interp1d
import json
import os

# Constants
G_SI = 6.67430e-11
C_SI = 2.99792458e8
M_SUN_KG = 1.98847e30

def ligo_design_sensitivity(f):
    """
    Approximate LIGO/Virgo/KAGRA design sensitivity curve.
    Returns noise ASD sqrt(S_n(f)) in dimensionless strain / sqrt(Hz).
    Simplified analytic fit for f in Hz.
    """
    # Approximate design curve
    f0 = 215.0  # Hz
    x = f / f0
    # Low-frequency wall
    S_low = 1e-46 * (100.0 / f)**4
    # Broad minimum
    S_mid = 1e-47 * (1.0 + x**2)
    # High-frequency rise
    S_high = 1e-47 * x**4
    S_n = S_low + S_mid + S_high
    return np.sqrt(S_n)

def generate_noise(td, fs, f_psd):
    """
    Generate stationary Gaussian noise with given one-sided PSD.
    """
    N = len(td)
    dt = td[1] - td[0]
    df = 1.0 / (N * dt)
    freqs = np.fft.rfftfreq(N, dt)

    # Interpolate PSD at positive frequencies
    psd_vals = f_psd(freqs)
    psd_vals = np.maximum(psd_vals, 1e-60)

    # Generate white noise in frequency domain
    white_real = np.random.normal(0, 1, len(freqs))
    white_imag = np.random.normal(0, 1, len(freqs))
    h_tilde = (white_real + 1j * white_imag) * np.sqrt(0.5 * psd_vals * N * df)

    # Inverse FFT to time domain
    noise = np.fft.irfft(h_tilde, n=N) / dt
    return noise

def matched_filter_snr(template_td, signal_td, td, f_psd):
    """
    Compute optimal matched-filter SNR.
    """
    dt = td[1] - td[0]
    N = len(td)
    freqs = np.fft.rfftfreq(N, dt)

    template_ft = np.fft.rfft(template_td) * dt
    signal_ft = np.fft.rfft(signal_td) * dt
    psd_vals = np.maximum(f_psd(freqs), 1e-60)

    # Inner product
    integrand = np.conj(template_ft) * signal_ft / psd_vals
    rho_sq = 4.0 * np.sum(integrand) * (freqs[1] - freqs[0])
    rho = np.real(rho_sq)

    # Normalize template
    norm_sq = 4.0 * np.sum(np.abs(template_ft)**2 / psd_vals) * (freqs[1] - freqs[0])
    snr = rho / np.sqrt(np.real(norm_sq))
    return float(np.real(snr))

def run_benchmark(M_solar, alpha, distance_mpc, num_trials=100):
    """
    Inject DPM echo signal into noise and compute SNR distribution.
    """
    from dpm_echo import solve_echo

    template = solve_echo(M_solar, alpha)
    t = np.array(template["time_s"])
    h_template = np.array(template["strain"])

    # Convert to detector frame strain amplitude: h ~ 10^-21 * (M/D)
    # D in Mpc
    amplitude_factor = 1e-21 * (M_solar / 30.0) * (400.0 / distance_mpc)
    h_template *= amplitude_factor

    # Uniformly sample time array
    fs = 4096.0  # Hz
    t_min, t_max = t.min(), t.max()
    N = int((t_max - t_min) * fs)
    t_uniform = np.linspace(t_min, t_max, N)
    h_uniform = interp1d(t, h_template, kind="cubic", fill_value=0,
                         bounds_error=False)(t_uniform)

    snrs = []
    for _ in range(num_trials):
        noise = generate_noise(t_uniform, fs, ligo_design_sensitivity)
        signal = h_uniform + noise
        snr = matched_filter_snr(h_uniform, signal, t_uniform,
                                  ligo_design_sensitivity)
        snrs.append(snr)

    return {
        "M_solar": float(M_solar),
        "alpha": float(alpha),
        "distance_mpc": float(distance_mpc),
        "num_trials": int(num_trials),
        "snr_mean": float(np.mean(snrs)),
        "snr_std": float(np.std(snrs)),
        "snr_max": float(np.max(snrs)),
        "detection_fraction": float(np.mean(np.array(snrs) > 8.0))
    }

def main():
    os.makedirs("data", exist_ok=True)

    benchmarks = []
    distances = [100, 200, 400, 800]
    configs = [(30.0, 1.0), (30.0, 2.0), (100.0, 1.0)]

    for M, alpha in configs:
        for D in distances:
            print(f"Running benchmark M={M}, alpha={alpha}, D={D} Mpc ...")
            res = run_benchmark(M, alpha, D, num_trials=50)
            benchmarks.append(res)
            print(f"  SNR mean={res['snr_mean']:.2f}, "
                  f"detection fraction={res['detection_fraction']:.2f}")

    with open("data/dpm_benchmark_results.json", "w") as f:
        json.dump(benchmarks, f, indent=2)

    # CSV summary
    import csv
    with open("data/dpm_benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M_solar", "alpha", "distance_mpc", "snr_mean",
                         "snr_std", "detection_fraction"])
        for res in benchmarks:
            writer.writerow([
                res["M_solar"], res["alpha"], res["distance_mpc"],
                f"{res['snr_mean']:.3f}",
                f"{res['snr_std']:.3f}",
                f"{res['detection_fraction']:.3f}"
            ])

    print("Saved data/dpm_benchmark_results.json and .csv")

if __name__ == "__main__":
    main()