# plot_diagnostics.py
"""
Re-generates training diagnostic figures used in the JMLR paper.
If the PNG already exists, it is skipped to avoid overwriting.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

figdir = Path("figures")
figdir.mkdir(exist_ok=True)

loss = np.load("loss_history.npy")
comp = np.load("loss_components.npz")

def save_fig(name):
    path = figdir / name
    if path.exists():
        print(f"[skip] {name} already present")
        return path
    return path

# -- 1) Total loss -------------------------------------------------
out = save_fig("fig_loss_curve.png")
if out:
    plt.figure(figsize=(4.5,3))
    plt.semilogy(loss)
    plt.xlabel("Epoch"); plt.ylabel("Total loss")
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()

# -- 2) train vs val data loss -------------------------------------
if "val_data" in comp:
    out = save_fig("fig_overfit.png")
    if out:
        plt.figure(figsize=(4.5,3))
        plt.semilogy(comp["data"], label="train")
        plt.semilogy(comp["val_data"], "--", label="validation")
        plt.xlabel("Epoch"); plt.ylabel("Data-loss")
        plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()

# -- 3) component breakdown ----------------------------------------
out = save_fig("fig_loss_components.png")
if out:
    plt.figure(figsize=(4.8,3))
    plt.semilogy(loss,           label="total")
    plt.semilogy(comp["data"],   label="data",   ls="--")
    plt.semilogy(comp["phys"],   label="physics",ls=":")
    plt.xlabel("Epoch"); plt.ylabel("Loss value")
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()

print("✓ plot_diagnostics.py finished.")
