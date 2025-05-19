#!/usr/bin/env python
# run_ablation.py  –  2×2 width × λ_phy experiment
import time, numpy as np, pandas as pd, torch
from pinn_sds import train_pinn, PINN, device, analytic_E

configs = [
    {"width":128, "lambda_phy":1},
    {"width":128, "lambda_phy":5},
    {"width":256, "lambda_phy":1},
    {"width":256, "lambda_phy":5},
]

rows = []
for cfg in configs:
    tag = f"W{cfg['width']}_L{cfg['lambda_phy']}"
    print(f"\n--- Training {tag} ---")
    t0 = time.time()
    model = train_pinn(width=cfg["width"],
                       λ_phy=cfg["lambda_phy"],
                       num_epochs=3000,       # hızlı deney
                       patience=600,
                       save_path=f"best_{tag}.pt")
    runtime = time.time() - t0

    # RMSE hesapla
    N = 4000
    r = torch.linspace(2.0, 150.0, N, device=device).view(-1,1)
    with torch.no_grad():
        rmse = np.sqrt(((model.net(r).cpu().numpy().squeeze()
                         - analytic_E(r).cpu().numpy().squeeze())**2).mean())

    rows.append({"Width": cfg["width"],
                 "λ_phy": cfg["lambda_phy"],
                 "RMSE": rmse,
                 "Runtime (s)": runtime})

df = pd.DataFrame(rows)
df.to_csv("table_T4_ablation.csv", index=False)
print("\n### Table T-4 — Ablation (width × λ_phy)")
print(df.to_markdown(index=False, floatfmt=".3e"))
