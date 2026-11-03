# DPM Simulations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Numerical simulations and benchmark tests for the **Dynamical Point Model (DPM)**:
a theoretical framework in which physical points carry internal quantum-geometric
structure, black-hole interiors host regular de-Sitter-like cores, and cosmic
structure grows through a hierarchy of chiral-quantum seeds.

## Author

**Xiaohui Li**  
Independent Researcher, Baoji, China  
Email: danceluck@163.com  
ORCID: 0009-0003-8233-7191

## Repository Structure
## ├── dpm_metric.py # Regular black-hole metric, horizon, Kretschmann scalar, volume ├── dpm_echo.py # Gravitational-wave echo templates ├── dpm_pbh_spectrum.py # Primordial black-hole mass spectrum ├── dpm_benchmark.py # Matched-filter benchmark with LIGO/Virgo/KAGRA sensitivity ├── data/ # Example outputs (JSON/CSV) ├── requirements.txt # Python dependencies ├── LICENSE # MIT License └── README.md # This file
## . ├── dpm_metric.py # Regular black-hole metric, horizon, Kretschmann scalar, volume ├── dpm_echo.py # Gravitational-wave echo templates ├── dpm_pbh_spectrum.py # Primordial black-hole mass spectrum ├── dpm_benchmark.py # Matched-filter benchmark with LIGO/Virgo/KAGRA sensitivity ├── data/ # Example outputs (JSON/CSV) ├── requirements.txt # Python dependencies ├── LICENSE # MIT License └── README.md # This file

Bash

## Installation

```bash
git clone https://github.com/xiaohui-li/dpm-simulations.git
cd dpm-simulations
pip install -r requirements.txt
Usage
Run each script independently:

Bash
python dpm_metric.py        # generates data/dpm_metric_examples.*
python dpm_echo.py          # generates data/echo_*.csv and echo_metadata.json
python dpm_pbh_spectrum.py  # generates data/dpm_pbh_spectrum.*
python dpm_benchmark.py     # generates data/dpm_benchmark_results.*
Physics Summary
The DPM regular metric is

f(r)=1−
frac2GMr 
2
 r 
3
 +2GMr 
c
2
​
 ,
where r 
c
​
 =
alpha
ell 
P
​
  is the quantum core radius. The central region approximates de Sitter space, replacing the classical Schwarzschild singularity with a finite, causally regular spacetime.

Key predictions:

Gravitational-wave echoes after the main ringdown, with time delay 
Deltat
approx8GM∣
ln
mathcalR∣.
Peaked primordial black-hole mass spectrum at asteroid and stellar mass scales.
High-energy cosmic-ray cutoff near 10 
20
  eV for stellar-mass black holes.
Data Files
File	Description
data/dpm_metric_examples.csv	Metric properties for sample black-hole masses
data/echo_metadata.json	Echo template metadata
data/echo_M*_alpha*.csv	Individual echo waveforms
data/dpm_pbh_spectrum.csv	PBH mass spectrum for several couplings
data/dpm_benchmark_results.csv	Matched-filter SNR benchmarks
Citation
If you use this code, please cite:

Li, X. The Mathematical Model of the Point: A Dynamical Spacetime Framework for Black-Hole Interiors (2025).

License
This project is released under the MIT License. See LICENSE for details.