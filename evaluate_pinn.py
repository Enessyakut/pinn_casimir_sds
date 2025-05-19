import os, numpy as np, matplotlib, torch, gc
matplotlib.use("Agg")                 # headless backend
import matplotlib.pyplot as plt
from pinn_sds import PINN, analytic_E

gc.collect()

# model (CPU)
model = PINN()
model.load_state_dict(torch.load("best_pinn.pt", map_location="cpu"))
model.eval()

# test grid
r = torch.linspace(2.0, 150.0, 2000).view(-1, 1)
with torch.no_grad():
    E_pred = model.net(r).squeeze()
E_ref = analytic_E(r).squeeze()

# numpy
r_np      = r.numpy().squeeze()
E_pred_np = E_pred.numpy()
E_ref_np  = E_ref.numpy()
rel_err   = np.abs(E_pred_np - E_ref_np) / (np.abs(E_ref_np) + 1e-12)

# energy figure
plt.figure()
plt.plot(r_np, E_ref_np, label="Analytic")
plt.plot(r_np, E_pred_np, "--", label="PINN")
plt.xlabel(r"$r$")
plt.ylabel(r"$E_C$")
plt.legend()
plt.tight_layout()
plt.savefig("fig_energy.png", dpi=300)
plt.close()

# relative-error figure
plt.figure()
plt.semilogy(r_np, rel_err)
plt.xlabel(r"$r$")
plt.ylabel("Relative Error")
plt.tight_layout()
plt.savefig("fig_relerr.png", dpi=300)
plt.close()

# save raw arrays
np.savez("pinn_results.npz",
         r=r_np, E_pred=E_pred_np, E_ref=E_ref_np, rel_err=rel_err)

print("✔ Figures saved and data written.")
