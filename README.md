# pinn_casimir_sds
Code, data and scripts for ‘Physics‐Informed Neural Networks for Casimir Energy in Schwarzschild–de Sitter Spacetime’ (Yakut 2025)
# Physics-Informed Neural Networks for Casimir Energy in SdS Spacetime
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)

Implementation that accompanies the paper

> **Yakut E.**  
> *Physics-Informed Neural Networks for Casimir Energy in Schwarzschild-de Sitter Spacetime*.  
> Journal of Machine Learning Research (JMLR), 2025.  
> [pre-print ↗](link-to-arXiv) • DOI: `10.XXXX/XXXX`

The repository contains **all code, data and figures** needed to fully
reproduce the results—training takes ≈10 min on a single CPU core.

---

## Directory layout
PINN_Project/
├── pinn_sds.py # model + training script
├── evaluate_pinn.py # generate fig_energy.png
├── plot_diagnostics.py # loss curves + over-fit check
├── make_error_plots.py # abs / scaled rel. error figures
├── residual_hist.py # residual histogram
├── loss_history.npy # saved by training
├── loss_components.npz # ” ”
├── figures/ # all figures used in the paper
└── tables/ # CSV tables T-1 … T-4
