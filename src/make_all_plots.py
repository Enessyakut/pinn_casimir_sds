#!/usr/bin/env python
# ----------------------------------------------------------
# Produce absolute error, scaled relative error and
# training loss curves in one shot
# ----------------------------------------------------------
import numpy as np, matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt

# ---------- 1) Enerji verilerini yükle ----------
dat = np.load("pinn_results.npz")
r        = dat["r"]
E_pred   = dat["E_pred"]
E_ref    = dat["E_ref"]
abs_err  = np.abs(E_pred - E_ref)

# ölçekli göreli hata (ε sabiti)
eps = 1e-6
scaled_rel_err = abs_err / (np.abs(E_ref) + eps)

# ---------- 2) Loss verisini yükle -------------
loss = np.load("loss_history.npy")
epochs = np.arange(1, len(loss)+1)

# ---------- 3) Mutlak hata grafiği -------------
plt.figure()
plt.semilogy(r, abs_err)
plt.xlabel(r"$r$")
plt.ylabel("Absolute Error")
plt.tight_layout()
plt.savefig("fig_abserr.png", dpi=300)
plt.close()

# ---------- 4) Ölçekli göreli hata -------------
plt.figure()
plt.semilogy(r, scaled_rel_err)
plt.xlabel(r"$r$")
plt.ylabel("Scaled Relative Error")
plt.tight_layout()
plt.savefig("fig_scaled_relerr.png", dpi=300)
plt.close()

# ---------- 5) Loss eğrisi ----------------------
plt.figure()
plt.semilogy(epochs, loss)
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("Training Loss Curve")
plt.tight_layout()
plt.savefig("fig_loss_curve.png", dpi=300)
plt.close()

print("✔ Plots saved: fig_abserr.png, fig_scaled_relerr.png, fig_loss_curve.png")
