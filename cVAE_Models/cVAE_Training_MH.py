import os
import cv2
import numpy as np
import torch
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)

import sys
sys.path.append('/path to codes/')
from Reflection_Structure_DataLoader import cVAEDataLoader
from Models import Autoencoder, cVAEtoStructureMH

autoencoder_model_path = "path to the autoencoder model/autoencoder_model_128_noise.pth"
ae_model = Autoencoder(latent_dim=128)
ae_model.load_state_dict(torch.load(autoencoder_model_path, map_location=DEVICE, weights_only=True))
ae_model.to(DEVICE)
ae_model.eval()

def get_latent_vector(ae_model, reflection_tensor):
    with torch.no_grad():
        _, c_latent = ae_model(reflection_tensor)
    return c_latent


def cvae_loss_function(x_recon, x, mu, logvar, loss_weights=None):
    if loss_weights is None:
        loss_weights = torch.ones(3, device=x.device)

    mse_per_dim = F.mse_loss(x_recon, x, reduction='none').mean(dim=0)  
    mae_per_dim = F.l1_loss(x_recon, x, reduction='none').mean(dim=0)   
    recon = torch.dot(loss_weights, mse_per_dim + mae_per_dim)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / x.size(0)  
    loss = recon + kl
    mse_loss = mse_per_dim.mean()
    mae_loss = mae_per_dim.mean()
    return loss, mse_loss, mae_loss, kl, mae_per_dim

lw = torch.tensor([1.0, 5.0, 1.0], device=DEVICE)

def train_cvae(model, train_loader, optimizer, device):
    model.train()
    total_loss, total_mse, total_mae, total_kl = 0, 0, 0, 0
    total_mae_vec = torch.zeros(3, device=DEVICE)
    for x, c in train_loader:

        x = x.to(device) #structure
        c = c.to(device) #reflection as condition
        c_latent = get_latent_vector(ae_model, c)       
        
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x, c_latent)
        loss, mse_loss, mae_loss, kl, mae_per_dim = cvae_loss_function(x_recon, x, mu, logvar, loss_weights=lw)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_mae += mae_loss.item()
        total_kl += kl.item()
        total_mae_vec += mae_per_dim.detach()
        
    n_batches = len(train_loader)
    return (total_loss / n_batches,
            total_mse / n_batches,
            total_mae / n_batches,
            total_kl / n_batches,
           total_mae_vec / n_batches)

def test_cvae(model, val_loader, device):
    model.eval()
    total_loss, total_mse, total_mae, total_kl = 0, 0, 0, 0
    total_mae_vec = torch.zeros(3, device=DEVICE)
    with torch.no_grad():
        for x, c in val_loader:
            x = x.to(device)
            c = c.to(device)
            c_latent = get_latent_vector(ae_model, c)          
            
            x_recon, mu, logvar = model(x, c_latent)
            loss, mse_loss, mae_loss, kl, mae_per_dim = cvae_loss_function(x_recon, x, mu, logvar, loss_weights=lw)
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_mae += mae_loss.item()
            total_kl += kl.item()
            total_mae_vec += mae_per_dim.detach()

    
    n_batches = len(val_loader)
    return (total_loss / n_batches,
            total_mse / n_batches,
            total_mae / n_batches,
            total_kl / n_batches,
           total_mae_vec / n_batches)

if __name__ == "__main__":

    data_loader = cVAEDataLoader(
        start_index = 0,
        end_index = 14999,
        folder_path="/path to data/",
        filename="Filenames_cvae_training_MH.txt"
    )

    structures, reflections, file_names = data_loader.load_data()
    reflections = np.expand_dims(reflections, axis=1) 

    structures_tensor = torch.tensor(structures, dtype=torch.float32)
    reflections_tensor = torch.tensor(reflections, dtype=torch.float32)

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    batch_size = 128
    num_epochs = 10000 
    patience = 700
    test_ratio = 0.2
    print(len(structures_tensor))
    
    trainval_idx, test_idx = train_test_split(np.arange(len(structures_tensor)), test_size=test_ratio, random_state=42)
    
    test_filenames = [file_names[i] for i in test_idx]
    test_filename_output = "Test_filenames_cVAE-AE_kfolds_optuna_training_MH.txt"
    with open(test_filename_output, "w") as f:
        for name in test_filenames:
            f.write(name + "\n")
    
    X_test = structures_tensor[test_idx]
    C_test = reflections_tensor[test_idx]
    
    test_dataset = TensorDataset(X_test, C_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(trainval_idx)):
        print(f"\n=== Fold {fold+1}/{k_folds} ===")

        train_idx = [trainval_idx[i] for i in train_idx]
        val_idx = [trainval_idx[i] for i in val_idx]

        val_filenames = [file_names[i] for i in val_idx]
        val_filename_output = f"Val_filenames_cVAE-AE_fold_{fold+1}_optuna_MH.txt"
        with open(val_filename_output, "w") as f:
            for name in val_filenames:
                f.write(name + "\n")
            
        X_train, X_val = structures_tensor[train_idx], structures_tensor[val_idx]
        C_train, C_val = reflections_tensor[train_idx], reflections_tensor[val_idx]

        train_dataset = TensorDataset(X_train, C_train)
        val_dataset = TensorDataset(X_val, C_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

        model = cVAEtoStructureMH(
            latent_dim=32,
            encoder_dropout=0.1,
            decoder_dropout=0.1
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.00075, weight_decay=5e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1, eta_min=1e-7)

        best_fold_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(num_epochs):
            train_loss, train_mse, train_mae, train_kl, train_mae_vec = train_cvae(model, train_loader, optimizer, DEVICE)
            val_loss, val_mse, val_mae, val_kl, val_mae_vec = test_cvae(model, val_loader, DEVICE)

            scheduler.step()
            
            print(f"[Fold {fold+1} | Epoch {epoch+1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.9f} | MSE: {train_mse:.9f} | MAE: {train_mae:.9f} | KL: {train_kl:.9f}")
            print(f"Val  Loss: {val_loss:.9f} | MSE: {val_mse:.9f} | MAE: {val_mae:.9f} | KL: {val_kl:.9f}")
            tm = train_mae_vec.detach().cpu().numpy()
            vm = val_mae_vec.detach().cpu().numpy()
            print(f"Ag, Ormo, Period Train: {np.round(tm, 4)}  Val: {np.round(vm, 4)}")


            if val_loss < best_fold_loss:
                best_fold_loss = val_loss
                epochs_no_improve = 0

                torch.save(model.state_dict(), f"cVAE-AE_model_fold_{fold+1}_optuna_MH.pth")
            else:
                if epoch >= 50:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch={epoch+1} for fold={fold+1}")
                        break

        fold_results.append(best_fold_loss)

    avg_loss = sum(fold_results) / len(fold_results)
    print("\nK-Fold results:")
    for i, result in enumerate(fold_results):
        print(f" fold {i+1}: Best test loss = {result:.9f}")
    print(f"Ave test toss across folds = {avg_loss:.9f}")
