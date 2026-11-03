#!/usr/bin/env python3
"""
dpm_metric.py
-------------
Compute the Dynamical Point Model (DPM) regular black-hole metric,
horizon locations, Kretschmann scalar, and internal volume.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import json
import os

# Natural units: G = c = 1
# Conversion factors (SI)
G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
C_SI = 2.99792458e8         # m/s
M_SUN_KG = 1.98847e30       # kg
L_PL_SI = 1.616255e-35      # Planck length in meters
T_PL_SI = 5.391245e-44      # Planck time in seconds

def si_to_geom(M_kg):
    """Convert SI mass (kg) to geometric units (length)."""
    return G_SI * M_kg / C_SI**2

def geom_to_si(M_geom):
    """Convert geometric length to SI mass (kg)."""
    return M_geom * C_SI**2 / G_SI

def planck_length(alpha=1.0):
    """DPM physical point/core radius: r_c = alpha * l_P."""
    return alpha * L_PL_SI

def metric_function(r, M_geom, r_c):
    """
    DPM regular metric function f(r) = 1 - 2 M r^2 / (r^3 + 2 M r_c^2).
    M_geom and r_c should be in geometric units (meters).
    """
    denom = r**3 + 2.0 * M_geom * r_c**2
    if denom <= 0:
        return 1.0
    return 1.0 - (2.0 * M_geom * r**2) / denom

def mass_function(r, M_geom, r_c):
    """
    Effective interior mass M(r) = M r^3 / (r^3 + 2 M r_c^2).
    """
    denom = r**3 + 2.0 * M_geom * r_c**2
    return M_geom * r**3 / denom

def find_horizons(M_geom, r_c):
    """
    Find roots of f(r) = 0.
    For M > M_ext, returns (inner_horizon, outer_horizon).
    For M = M_ext, returns (extremal_horizon,).
    For M < M_ext, returns empty tuple.
    """
    # Search range
    r_max = max(10.0 * M_geom, 1e-30)
    r_min = max(r_c * 1e-3, 1e-50)

    rs = np.logspace(np.log10(r_min), np.log10(r_max), 20000)
    fs = np.array([metric_function(r, M_geom, r_c) for r in rs])

    roots = []
    for i in range(len(fs) - 1):
        if fs[i] == 0.0:
            roots.append(rs[i])
        elif fs[i] * fs[i + 1] < 0:
            root = brentq(
                lambda x: metric_function(x, M_geom, r_c),
                rs[i], rs[i + 1], xtol=1e-30
            )
            roots.append(root)

    # Remove duplicates and sort
    roots = sorted(list(set([round(r, 30) for r in roots])))
    return tuple(roots)

def kretschmann(r, M_geom, r_c):
    """
    Kretschmann scalar K = R_{abcd} R^{abcd} for the metric (3)-(4).
    Derived from the Riemann tensor of a spherically symmetric static metric.
    """
    f = metric_function(r, M_geom, r_c)
    m = mass_function(r, M_geom, r_c)

    # First derivative of f: f' = -2 M(r)' / r + 2 M(r) / r^2
    # But easier to compute analytically:
    denom = r**3 + 2.0 * M_geom * r_c**2
    dMdr = 3.0 * M_geom * r**2 * (r**3 + 2.0 * M_geom * r_c**2)
    dMdr -= M_geom * r**3 * 3.0 * r**2
    dMdr /= denom**2

    fp = -2.0 * dMdr / r + 2.0 * m / r**2
    fpp = (2.0 * dMdr / r**2
           - 4.0 * m / r**3
           + 6.0 * M_geom * r_c**2 * r / denom**2)  # approximate second term

    # Kretschmann for metric ds^2 = -f dt^2 + f^{-1} dr^2 + r^2 dOmega^2
    K = (fpp**2 / 4.0
         + fp**2 / (2.0 * r**2)
         + (1.0 - f)**2 / r**4)
    return 4.0 * K  # factor from spherical symmetry

def internal_volume(M_geom, r_c, r_h):
    """
    Internal volume V_int = 4*pi*int_0^{r_h} r^2 / sqrt(f(r)) dr.
    """
    integrand = lambda r: r**2 / np.sqrt(max(metric_function(r, M_geom, r_c), 1e-30))
    V, _ = quad(integrand, 0, r_h, limit=100)
    return 4.0 * np.pi * V

def compute_black_hole(M_solar, alpha=1.0):
    """
    Convenience wrapper: input mass in solar masses, return DPM properties.
    """
    M_kg = M_solar * M_SUN_KG
    M_geom = si_to_geom(M_kg)
    r_c = planck_length(alpha)

    horizons = find_horizons(M_geom, r_c)

    result = {
        "M_solar": float(M_solar),
        "alpha": float(alpha),
        "M_geom_m": float(M_geom),
        "r_c_m": float(r_c),
        "r_c_over_r_h": None,
        "horizons_m": [float(h) for h in horizons],
        "kretschmann_center": float(kretschmann(r_c * 0.1, M_geom, r_c)),
        "internal_volume_m3": None,
    }

    if len(horizons) >= 2:
        r_h = horizons[-1]  # outer horizon
        r_in = horizons[0]
        result["r_h_m"] = float(r_h)
        result["r_in_m"] = float(r_in)
        result["r_c_over_r_h"] = float(r_c / r_h)
        result["internal_volume_m3"] = float(internal_volume(M_geom, r_c, r_h))
    elif len(horizons) == 1:
        result["r_h_m"] = float(horizons[0])
        result["internal_volume_m3"] = float(internal_volume(M_geom, r_c, horizons[0]))
    else:
        result["r_h_m"] = None

    return result

def main():
    os.makedirs("data", exist_ok=True)

    masses = np.array([1.0, 10.0, 30.0, 100.0, 1e6])
    alphas = np.array([0.5, 1.0, 2.0])

    results = []
    for M in masses:
        for alpha in alphas:
            res = compute_black_hole(M, alpha)
            results.append(res)

    # Save detailed JSON
    with open("data/dpm_metric_examples.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save concise CSV
    import csv
    with open("data/dpm_metric_examples.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "M_solar", "alpha", "r_c_m", "r_h_m", "r_c/r_h",
            "K_center", "V_int_m3"
        ])
        for res in results:
            writer.writerow([
                res["M_solar"],
                res["alpha"],
                f"{res['r_c_m']:.3e}",
                f"{res.get('r_h_m', 'N/A')}",
                f"{res.get('r_c_over_r_h', 'N/A')}",
                f"{res['kretschmann_center']:.3e}",
                f"{res.get('internal_volume_m3', 'N/A')}"
            ])

    print("Saved data/dpm_metric_examples.json and data/dpm_metric_examples.csv")
    for res in results[:3]:
        print(f"M={res['M_solar']} M⊙, alpha={res['alpha']}: "
              f"r_h={res.get('r_h_m', 'N/A')} m, "
              f"r_c/r_h={res.get('r_c_over_r_h', 'N/A')}")

if __name__ == "__main__":
    main()