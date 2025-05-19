#!/usr/bin/env python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dat = np.load("loss_components.npz")
epochs = np.arange(1, len(dat["total"])+1)

plt.figure()
plt.semilogy(epochs, dat["total"], label="Total")
plt.semilogy(epochs, dat["data"],  label="Data")
plt.semilogy(epochs, dat["phys"],  label="Physics")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PINN Training Loss Components")
plt.legend()
plt.tight_layout()
plt.savefig("fig_loss_components.png", dpi=300)
plt.close()

print("✔ Loss-component plot saved: fig_loss_components.png")
