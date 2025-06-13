# -------------------------------------------------------------
# pinn_sds.py • PINN modeli + eğitim betiği
# Bellek-dostu (checkpointing) – torch.compile devre dışı
# -------------------------------------------------------------
import math, warnings, time, numpy as np
import torch, torch.nn as nn
from torch import amp
from contextlib import nullcontext

# ---------- 0) Cihaz & rastgelelik ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)
torch.set_default_dtype(torch.float32)

loss_history   = []
data_loss_hist = []
phys_loss_hist = []
val_loss_hist  = []

# ---------- DummyScaler ----------
class DummyScaler:
    def scale(self, x): return x
    def step(self, opt): opt.step()
    def update(self):     pass

# ---------- compile güvenli sarmalayıcı ----------
def safe_compile(model, **_):
    warnings.warn("[torch.compile disabled → using eager mode]")
    return model

# ---------- Fizik sabitleri ----------
M, Lambda, L_scale, C_const = 1.0, 1e-52, 50.0, 1.0
def f_metric(r): return 1.0 - 2*M/r - (Lambda*r**2)/3
def g_func(r):   return (C_const/r**4) * torch.exp(-2*r/L_scale)
def analytic_E(r, a=1.0): return (1/r**2) * torch.exp(-r/a)

# ---------- Aktivasyon & katman ----------
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)
class FCBlock(nn.Module):
    def __init__(s, i, o):
        super().__init__(); s.lin = nn.Linear(i, o)
        nn.init.xavier_uniform_(s.lin.weight)
    def forward(s, x): return Swish()(s.lin(x))

# ---------- PINN ----------
class PINN(nn.Module):
    def __init__(s, layers=4, width=128):
        super().__init__()
        net = [nn.Linear(1, width), Swish()]
        for _ in range(layers-1): net.append(FCBlock(width, width))
        net.append(nn.Linear(width, 1))
        s.net = nn.Sequential(*net)
    def forward(s, r):
        r = r.requires_grad_(True)
        E = s.net(r)
        dE_dr  = torch.autograd.grad(E, r, torch.ones_like(E), create_graph=True)[0]
        d2E_dr = torch.autograd.grad(dE_dr, r, torch.ones_like(dE_dr), create_graph=True)[0]
        return E, d2E_dr + f_metric(r)*dE_dr + g_func(r)*E

# ---------- Eğitim ----------
def train_pinn(layers=4, width=128, lr=1e-3, num_epochs=5000,
               patience=600, λ_phy=1.0, batch_size=512,
               N_colloc=4000, save_path="best_pinn.pt"):

    model = safe_compile(PINN(layers, width).to(device))

    r_col = torch.linspace(2*M, 150.0, N_colloc, device=device).view(-1, 1)
    val_mask = torch.randperm(N_colloc, device=device)[: N_colloc // 10]
    r_val = r_col[val_mask].detach()

    mse   = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=500)

    if torch.cuda.is_available():
        scaler, autocast_ctx = amp.GradScaler(), amp.autocast(device_type="cuda")
    else:
        scaler, autocast_ctx = DummyScaler(), nullcontext()

    accum = N_colloc // batch_size
    best_loss, wait, t0 = math.inf, 0, time.time()

    for ep in range(1, num_epochs+1):
        opt.zero_grad(set_to_none=True)
        tot=data=phy=0.0
        idx = torch.randperm(N_colloc, device=device)

        for i in range(accum):
            r_b = r_col[idx[i*batch_size:(i+1)*batch_size]].clone().detach().requires_grad_(True)
            with autocast_ctx:
                E_b, ode_b = model(r_b)
                L_phy = mse(ode_b, torch.zeros_like(ode_b))
                L_dat = mse(E_b, analytic_E(r_b).detach())
                loss  = L_dat + λ_phy * L_phy
                scaler.scale(loss/accum).backward()
            tot+=loss.item(); data+=L_dat.item(); phy+=L_phy.item()

        scaler.step(opt); scaler.update(); sched.step(tot)

        # --- validation loss (data term) ---
        with torch.no_grad():
            val_loss = mse(model.net(r_val), analytic_E(r_val))

        # --- kayıtlar her zaman birlikte ---
        loss_history.append(tot)
        data_loss_hist.append(data)
        phys_loss_hist.append(phy)
        val_loss_hist.append(val_loss.item())

        if tot < best_loss:
            best_loss, wait = tot, 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
        if wait > patience:
            print(f"Early stop @ {ep}")
            break
        if ep % 500 == 0 or ep == 1:
            print(f"Ep {ep:4d} Loss {tot:.3e} (data {data:.2e}, phys {phy:.2e})  Val {val_loss:.2e}")

    # --- uzunluk eşitle ---
    min_len = min(len(loss_history), len(val_loss_hist))
    loss_history[:]   = loss_history[:min_len]
    data_loss_hist[:] = data_loss_hist[:min_len]
    phys_loss_hist[:] = phys_loss_hist[:min_len]
    val_loss_hist[:]  = val_loss_hist[:min_len]

    # --- kaydet ---
    np.save("loss_history.npy", np.array(loss_history))
    np.save("val_loss_hist.npy", np.array(val_loss_hist))
    np.savez("loss_components.npz",
             total=np.array(loss_history),
             data=np.array(data_loss_hist),
             phys=np.array(phys_loss_hist))

    print(f"Süre {time.time()-t0:.1f}s   En iyi loss {best_loss:.3e}")
    return model

# ---------- main ----------
if __name__ == "__main__":
    train_pinn()
