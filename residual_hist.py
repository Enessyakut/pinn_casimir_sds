#!/usr/bin/env python
# --------------------------------------------
# Plot histogram of ODE residuals R(r)
# --------------------------------------------
import torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pinn_sds import PINN, f_metric, g_func, device

# 1) yükle model (CPU)
model = PINN().to(device)
model.load_state_dict(torch.load("best_pinn.pt", map_location=device))
model.eval()

# 2) aynı kollokasyon ızgarası (eğitimle tutarlı)
N_colloc = 4000
r = torch.linspace(2.0, 150.0, N_colloc, device=device).view(-1, 1).requires_grad_(True)

# grad takibi açık; model parametrelerine grad gerekmediğinden
# no_grad kullanmıyoruz

               # doğrudan ağ çıktısı
E = model.net(r)

# --- türevler ---
dE_dr  = torch.autograd.grad(
            E, r, torch.ones_like(E), create_graph=True)[0]   # <-- create_graph=True

d2E_dr = torch.autograd.grad(
            dE_dr, r, torch.ones_like(dE_dr), create_graph=False)[0]

R = d2E_dr + f_metric(r)*dE_dr + g_func(r)*E      # residual tensor
R_np = R.detach().squeeze().cpu().numpy()


# 3) histogram (log |R|)
abs_R = np.abs(R_np) + 1e-30                     # sıfır korunması
plt.figure()
plt.hist(np.log10(abs_R), bins=60, color="blue", alpha=0.8)
plt.xlabel(r"$\log_{10}|R(r)|$")
plt.ylabel("Count")
plt.title("Histogram of ODE Residuals")
plt.tight_layout()
plt.savefig("fig_residual_hist.png", dpi=300)
plt.close()

print("✔ Residual histogram saved: fig_residual_hist.png")
