import torch, numpy as np, matplotlib.pyplot as plt
from src.pinn_sds import PINN, analytic_E, device

MODEL = "data/best_pinn.pt"
N = 2000

net = PINN().to(device)
net.load_state_dict(torch.load(MODEL, map_location=device))
net.eval()

r = torch.linspace(2.0, 150.0, N, device=device).view(-1, 1)
with torch.no_grad():
    E_pred, _ = net(r)
E_true = analytic_E(r)

abs_err = (E_true - E_pred).abs().cpu().numpy()
scaled  = (r.cpu().numpy()**2) * abs_err / (E_true.abs().cpu().numpy() + 1e-12)

plt.figure(figsize=(4,3))
plt.loglog(r.cpu(), abs_err)
plt.xlabel("$r$"); plt.ylabel("Absolute Error")
plt.tight_layout(); plt.savefig("figures/fig_abserr.png", dpi=300)

plt.figure(figsize=(4,3))
plt.loglog(r.cpu(), scaled)
plt.xlabel("$r$"); plt.ylabel(r"$r^{2}\times$Relative Error")
plt.tight_layout(); plt.savefig("figures/fig_scaled_relerr.png", dpi=300)
