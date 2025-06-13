#!/usr/bin/env python
import numpy as np, pandas as pd

dat = np.load("pinn_results.npz")          # evaluate_pinn.py çıktısı
r, E_pred, E_ref = dat["r"], dat["E_pred"], dat["E_ref"]
abs_err  = np.abs(E_pred - E_ref)
eps = 1e-6
scaled_rel_err = abs_err / (np.abs(E_ref) + eps)

# ---- T-1 error stats ----
err_stats = {
    "Metric": ["Abs. Error", "Scaled Rel. Error"],
    "Max":    [abs_err.max(), scaled_rel_err.max()],
    "Mean":   [abs_err.mean(), scaled_rel_err.mean()],
    "RMSE":   [np.sqrt((abs_err**2).mean()),
               np.sqrt((scaled_rel_err**2).mean())],
}
df_err = pd.DataFrame(err_stats).set_index("Metric")
print("### Table T-1 — Error Statistics\n")
print(df_err.to_markdown(floatfmt=".3e"), "\n")

# ---- T-2 pointwise compare ----
points = np.array([2.1, 2.2, 2.5, 3.0, 100, 120, 150])   # in units of M
rows = []
for p in points:
    idx = np.argmin(np.abs(r - p))
    rows.append({
        "r (M units)":    p,
        "E_C analytic":   E_ref[idx],
        "E_C PINN":       E_pred[idx],
        "Abs. Err":       abs_err[idx],
    })
df_pts = pd.DataFrame(rows)
print("### Table T-2 — Pointwise Energy Comparison\n")
print(df_pts.to_markdown(floatfmt=".3e"))

# ---- write optional CSVs ----
df_err.to_csv("table_T1_error_stats.csv")
df_pts.to_csv("table_T2_pointwise.csv")
