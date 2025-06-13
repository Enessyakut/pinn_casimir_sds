#!/usr/bin/env python
# -----------------------------------------------
# Generate absolute and scaled relative error plots
# -----------------------------------------------
import numpy as np, matplotlib
matplotlib.use("Agg")           # headless
import matplotlib.pyplot as plt

# 1) Load data saved by evaluate_pinn.py
data = np.load("pinn_results.npz")
r        = data["r"]
E_pred   = data["E_pred"]
E_ref    = data["E_ref"]
abs_err  = np.abs(E_pred - E_ref)

# 2) Scaled relative error  (ε avoids division by ~0)
eps            = 1e-6           # choose ε = 1×10⁻⁶ (global constant)
scaled_rel_err = abs_err / (np.abs(E_ref) + eps)

# 3) Absolute error plot  (log–y)
plt.figure()
plt.semilogy(r, abs_err)
plt.xlabel(r"$r$")
plt.ylabel("Absolute Error")
plt.tight_layout()
plt.savefig("fig_abserr.png", dpi=300)
plt.close()

# 4) Scaled relative error plot
plt.figure()
plt.semilogy(r, scaled_rel_err)
plt.xlabel(r"$r$")
plt.ylabel("Scaled Relative Error")
plt.tight_layout()
plt.savefig("fig_scaled_relerr.png", dpi=300)
plt.close()

print("✔ Error plots saved: fig_abserr.png, fig_scaled_relerr.png")
