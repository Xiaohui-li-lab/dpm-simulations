#!/usr/bin/env python3
"""
dpm_pbh_spectrum.py
-------------------
Compute the primordial black-hole mass spectrum predicted by the DPM.
"""

import numpy as np
import json
import os

def pbh_mass_spectrum(M, M_star, sigma, A=1.0):
    """
    Log-normal mass function:
    dn/dlnM = A * exp[-(ln M - ln M_*)^2 / (2 sigma^2)]
    M and M_star in solar masses.
    """
    lnM = np.log(M)
    lnMs = np.log(M_star)
    return A * np.exp(-0.5 * ((lnM - lnMs) / sigma) ** 2)

def dpm_pbh_spectrum(M, g1=0.1, A=1.0):
    """
    DPM prediction: two characteristic peaks.
    Peak 1: asteroid-mass PBHs (M_* ~ 1e-12 M_sun)
    Peak 2: stellar-mass PBHs (M_* ~ 30 M_sun)
    Relative amplitude controlled by g1.
    """
    # Peak positions
    M1_star = 1e-12
    M2_star = 30.0

    # Widths
    sigma1 = 2.0
    sigma2 = 1.5

    # Relative amplitude: stellar peak suppressed for large g1
    A2 = 0.5 / (1.0 + g1)

    return (pbh_mass_spectrum(M, M1_star, sigma1, A)
            + pbh_mass_spectrum(M, M2_star, sigma2, A * A2))

def main():
    os.makedirs("data", exist_ok=True)

    M = np.logspace(-15, 4, 1000)  # solar masses

    g1_values = [0.05, 0.1, 0.5, 1.0]

    results = {"masses": M.tolist()}
    for g1 in g1_values:
        spectrum = dpm_pbh_spectrum(M, g1=g1)
        results[f"g1_{g1}"] = spectrum.tolist()

    # Save JSON
    with open("data/dpm_pbh_spectrum.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    header = "M_solar," + ",".join([f"g1_{g1}" for g1 in g1_values])
    data = np.column_stack([M] + [dpm_pbh_spectrum(M, g1=g1) for g1 in g1_values])
    np.savetxt("data/dpm_pbh_spectrum.csv", data, delimiter=",",
               header=header, comments="")

    print("Saved data/dpm_pbh_spectrum.json and data/dpm_pbh_spectrum.csv")
    for g1 in g1_values:
        spectrum = dpm_pbh_spectrum(M, g1=g1)
        idx_max = np.argmax(spectrum)
        print(f"g1={g1}: peak at M={M[idx_max]:.2e} M⊙, "
              f"max dn/dlnM={spectrum[idx_max]:.3e}")

if __name__ == "__main__":
    main()