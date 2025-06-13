#!/usr/bin/env python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

train = np.load("loss_history.npy")
val   = np.load("val_loss_hist.npy")
epochs = np.arange(1, len(train)+1)

plt.figure()
plt.semilogy(epochs, train, label="Training Data-Loss")
plt.semilogy(epochs, val,   label="Validation Data-Loss")
plt.xlabel("Epoch"); plt.ylabel("MSE (Data term)")
plt.title("Over-fitting Check: Training vs Validation")
plt.legend(); plt.tight_layout()
plt.savefig("fig_overfit.png", dpi=300)
plt.close()

print("✔ Over-fitting plot saved: fig_overfit.png")
