#!/usr/bin/env python
# --------------------------------------
# Plot training loss vs epoch
# --------------------------------------
import numpy as np, matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt

loss = np.load("loss_history.npy")
epochs = np.arange(1, len(loss)+1)

plt.figure()
plt.semilogy(epochs, loss)
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("Training Loss Curve")
plt.tight_layout()
plt.savefig("fig_loss_curve.png", dpi=300)
plt.close()

print("✔ Loss curve saved: fig_loss_curve.png")
