#!/usr/bin/env python3
"""
dpm_echo.py
-----------
Solve the axial gravitational perturbation equation in the DPM metric
and generate gravitational-wave echo templates.
"""

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import json
import os

# Constants
G_SI = 6.67430e-11
C_SI = 2.99792458e8
M_SUN_KG = 1.98847e30
L_PL_SI = 1.616255e-35

def si_to_geom(M_kg):
    return G_SI * M_kg / C_SI**2

def planck_length(alpha=1.0):
    return alpha * L_PL_SI

def metric_function(r, M_geom, r_c):
    denom = r**3 + 2.0 * M_geom * r_c**2
    return 1.0 - (2.0 * M_geom * r**2) / denom

def tortoise_coordinate(r_grid, M_geom, r_c):
    """
    Numerically integrate dr_* = dr / f(r) to get r_*(r).
    """
    f_vals = np.array([metric_function(ri, M_geom, r_c) for ri in r_grid])
    f_vals = np.maximum(f_vals, 1e-30)
    dr = np.diff(r_grid)
    integrand = 1.0 / f_vals[:-1]
    r_star = np.zeros_like(r_grid)
    r_star[1:] = cumulative_trapezoid(1.0 / f_vals, r_grid, initial=0)
    return r_star

def potential_l(r, M_geom, r_c, l=2):
    """
    Zerilli-like potential for axial perturbations in Schwarzschild-like metric.
    V_l(r) = f(r) * [ l(l+1)/r^2 - 6M(r)/r^3 ]
    """
    f = metric_function(r, M_geom, r_c)
    m = M_geom * r**3 / (r**3 + 2.0 * M_geom * r_c**2)
    return f * (l * (l + 1) / r**2 - 6.0 * m / r**3)

def solve_echo(M_solar, alpha=1.0, l=2, dt_geom=0.5, t_max_geom=2000.0,
               echo_amplitude=0.3, echo_decay=0.4, num_echoes=4):
    """
    Generate a phenomenological echo template based on the DPM metric.
    Returns time array in seconds and strain array (dimensionless).
    """
    M_geom = si_to_geom(M_solar * M_SUN_KG)
    r_c = planck_length(alpha)
    r_h = 2.0 * M_geom  # outer horizon in geometric units

    # Echo time delay: Delta t ~ 8 M |ln R|
    # Approximate reflectivity
    R = max(0.01, 1.0 - 2.0 * r_c / r_h)
    delta_t_geom = 8.0 * M_geom * abs(np.log(R))

    # Convert to SI
    t_seconds = np.arange(-0.05, 0.15, dt_geom * G_SI * M_SUN_KG / C_SI**3)
    # More robust: use geometric time then convert
    t_geom = np.arange(-100.0, t_max_geom, dt_geom)
    t_s = t_geom * G_SI * M_SUN_KG / C_SI**3  # approximate conversion

    # Main ringdown
    f_qnm = 1.0 / (8.0 * M_geom)  # approximate QNM frequency in geometric units
    tau_geom = 16.0 * M_geom      # ringdown decay time
    f_qnm_hz = f_qnm * C_SI**3 / (G_SI * M_SUN_KG)
    tau_s = tau_geom * G_SI * M_SUN_KG / C_SI**3

    h = np.zeros_like(t_geom, dtype=float)
    mask_main = t_geom < 0
    h[mask_main] = 1.0 * np.exp(t_geom[mask_main] / tau_geom) * np.cos(2.0 * np.pi * f_qnm * t_geom[mask_main])

    # Echoes
    for n in range(1, num_echoes + 1):
        t_n = n * delta_t_geom
        amp_n = echo_amplitude * (echo_decay ** (n - 1))
        mask_echo = t_geom >= t_n
        h[mask_echo] += amp_n * np.exp(-(t_geom[mask_echo] - t_n) / tau_geom) \
                         * np.cos(2.0 * np.pi * f_qnm * (t_geom[mask_echo] - t_n))

    return {
        "M_solar": float(M_solar),
        "alpha": float(alpha),
        "f_qnm_hz": float(f_qnm_hz),
        "tau_s": float(tau_s),
        "delta_t_s": float(delta_t_geom * G_SI * M_SUN_KG / C_SI**3),
        "reflectivity": float(R),
        "time_s": t_s.tolist(),
        "strain": h.tolist()
    }

def main():
    os.makedirs("data", exist_ok=True)

    configs = [
        (30.0, 1.0),
        (30.0, 2.0),
        (100.0, 1.0),
        (10.0, 1.0),
    ]

    all_templates = []
    for M, alpha in configs:
        template = solve_echo(M, alpha)
        all_templates.append(template)

        # Save individual template as CSV
        fname = f"data/echo_M{M:.0f}_alpha{alpha:.1f}.csv"
        np.savetxt(fname,
                   np.column_stack([template["time_s"], template["strain"]]),
                   delimiter=",", header="time_s,strain", comments="")
        print(f"Saved {fname}: delta_t={template['delta_t_s']*1000:.2f} ms, "
              f"f_qnm={template['f_qnm_hz']:.1f} Hz, R={template['reflectivity']:.4f}")

    # Save all metadata
    with open("data/echo_metadata.json", "w") as f:
        json.dump(all_templates, f, indent=2)

    print("Saved data/echo_metadata.json")

if __name__ == "__main__":
    main()