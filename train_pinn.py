# train_pinn.py
from src.pinn_sds import train_pinn

if __name__ == "__main__":
    train_pinn(save_path="data/best_pinn.pt")
