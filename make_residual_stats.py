#!/usr/bin/env python
# make_residual_stats.py  –  Table T-3 generator
import numpy as np, torch
from pinn_sds import PINN, f_metric, g_func, device

# ----- modeli yükle -----
model = PINN().to(device)
model.load_state_dict(torch.load("best_pinn.pt", map_location=device))
model.eval()

# ----- residual hesapla -----
N = 4000
r = torch.linspace(2.0, 150.0, N, device=device).view(-1,1).requires_grad_(True)
E = model.net(r)
dE_dr  = torch.autograd.grad(E, r, torch.ones_like(E), create_graph=True)[0]
d2E_dr = torch.autograd.grad(dE_dr, r, torch.ones_like(dE_dr), create_graph=True)[0]
R = (d2E_dr + f_metric(r)*dE_dr + g_func(r)*E).detach().cpu().abs().numpy()

stats = {
    "min":    np.min(R),
    "median": np.median(R),
    "p95":    np.percentile(R, 95),
    "max":    np.max(R),
}
print("### Table T-3 — Residual Statistics (|R|)")
for k, v in stats.items():
    print(f"{k:>6}: {v:.3e}")

np.save("R_values.npy", R)      # opsiyonel veri kaydı
