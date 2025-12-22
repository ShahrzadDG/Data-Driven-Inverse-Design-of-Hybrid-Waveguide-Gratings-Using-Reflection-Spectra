import os
import cv2
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

random.seed(16)
np.random.seed(16)
torch.manual_seed(16)
torch.cuda.manual_seed(16)
torch.cuda.manual_seed_all(16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import sys
sys.path.append('/path to code/')
from Reflection_Structure_DataLoader import TandemDataLoader
from Models import Autoencoder, StructureToLatent, LatenttoStructureMH, TandemNetwork

def load_structure_to_latent_model(model_path, input_dim=3, output_dim=128, device='cpu'):
    model = StructureToLatent(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def structure_loss(pred_x, true_x, loss_weights: torch.Tensor):
    mse_per_dim = F.mse_loss(pred_x, true_x, reduction='none').mean(dim=0)  
    mae_per_dim = F.l1_loss(pred_x, true_x, reduction='none').mean(dim=0)   
    recon = torch.dot(loss_weights, mse_per_dim + mae_per_dim)
    mse_mean = mse_per_dim.mean()
    mae_mean = mae_per_dim.mean()
    return recon, mse_mean, mae_mean, mae_per_dim

lw = torch.tensor([1.0, 5.0, 1.0], device=device)  
str_weight = 1.0  
def train_TandemModel(model, train_loader, device='cpu'):
    model.inverse_model.train()
    model.forward_model.eval()
    total_latent_mse = 0.0
    total_struct_mse = 0.0
    total_struct_mae = 0.0
    total_mae_vec = torch.zeros(3, device=device)
    total_samples = 0
    for x, c in train_loader:

        x = x.to(device) #structure
        c = c.to(device) #reflection, latent as condition

        optimizer.zero_grad()
        latent_pred = model(c)
        latent_loss = criterion(latent_pred, c)

        struct_pred = model.pred(c)         
        struct_loss, struct_mse, struct_mae, mae_per_dim = structure_loss(struct_pred, x, lw)
        total_loss = latent_loss + str_weight * struct_loss

        total_loss.backward()
        optimizer.step()
        
        total_latent_mse += latent_loss.item()
        total_struct_mse += struct_mse.item()
        total_struct_mae += struct_mae.item()
        total_mae_vec += mae_per_dim.detach()

    n_batches = len(train_loader)
    return (total_latent_mse / n_batches,
            total_struct_mse / n_batches,
            total_struct_mae / n_batches,
            total_mae_vec / n_batches)

def test_TandemModel(model, val_loader, device):
    model.eval()
    total_latent_mse = 0.0
    total_struct_mse = 0.0
    total_struct_mae = 0.0
    total_mae_vec = torch.zeros(3, device=device)
    total_samples = 0
    with torch.no_grad():
        for x, c in val_loader:
            x = x.to(device)
            c = c.to(device)
            outputs = model(c)
            latent_loss = criterion(outputs, c)
            struct_pred = model.pred(c)     
            struct_loss, struct_mse, struct_mae, mae_per_dim = structure_loss(struct_pred, x, lw)
            
            total_latent_mse += latent_loss.item()
            total_struct_mse += struct_mse.item()
            total_struct_mae += struct_mae.item()
            total_mae_vec += mae_per_dim
    n_batches = len(val_loader)
    return (total_latent_mse / n_batches,
            total_struct_mse / n_batches,
            total_struct_mae / n_batches,
            total_mae_vec / n_batches)

if __name__ == "__main__":

    data_loader = TandemDataLoader(
        start_index=0,
        end_index=14999,
        folder_path="/path to data/"
    )

    structures, reflections, file_names = data_loader.load_data()

    structures_tensor = torch.tensor(structures, dtype=torch.float32)
    reflections_tensor = torch.tensor(reflections, dtype=torch.float32)

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    batch_size = 256
    num_epochs = 10000
    patience = 700
    test_ratio = 0.2

    trainval_idx, test_idx = train_test_split(np.arange(len(structures_tensor)), test_size=test_ratio, random_state=42)
    
    test_filenames = [file_names[i] for i in test_idx]
    test_filename_output = "Test_filenames_Tandem_training_MH.txt"
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
        val_filename_output = f"Val_filenames_Tandem_fold_{fold+1}_training_MH.txt"
        with open(val_filename_output, "w") as f:
            for name in val_filenames:
                f.write(name + "\n")
            
        X_train, X_val = structures_tensor[train_idx], structures_tensor[val_idx]
        C_train, C_val = reflections_tensor[train_idx], reflections_tensor[val_idx]

        train_dataset = TensorDataset(X_train, C_train)
        val_dataset = TensorDataset(X_val, C_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

        forward_model = load_structure_to_latent_model("/path to model/Forward_model_Strtolatent_optuna_NoisyAE.pth", device=device)
        for param in forward_model.parameters():
            param.requires_grad = False
        
        inverse_model = LatenttoStructureMH(input_dim=128, output_dim=3).to(device)      
        model = TandemNetwork(forward_model, inverse_model).to(device)
        
        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.AdamW(model.inverse_model.parameters(), lr=0.0004, weight_decay=1e-4)
        #scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-7)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1, eta_min=1e-7)

        best_fold_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(num_epochs):
            tr_lat_mse, tr_x_mse, tr_x_mae, tr_mae_vec = train_TandemModel(model, train_loader, device=device)
            va_lat_mse, va_x_mse, va_x_mae, va_mae_vec = test_TandemModel(model, val_loader, device=device)

            scheduler.step()
            
            tm = tr_mae_vec.detach().cpu().tolist()
            vm = va_mae_vec.detach().cpu().tolist()
            
            print(f"[Fold {fold+1} | Epoch {epoch+1}/{num_epochs}]")
            print(f"Train Latent MSE: {tr_lat_mse:.6f} | Struct MSE: {tr_x_mse:.6f} | Struct MAE: {tr_x_mae:.6f} | MAE per-dim (S,O,P): {tm[0]:.6f}, {tm[1]:.6f}, {tm[2]:.6f}")
            print(f"Val Latent MSE: {va_lat_mse:.6f} | Struct MSE: {va_x_mse:.6f} | Struct MAE: {va_x_mae:.6f} | MAE per-dim (S,O,P): {vm[0]:.6f}, {vm[1]:.6f}, {vm[2]:.6f}")


            for param_group in optimizer.param_groups:
                print(f"Learning Rate: {param_group['lr']:.10f}")
            
            if va_lat_mse < best_fold_loss:
                best_fold_loss = va_lat_mse 
                epochs_no_improve = 0

                torch.save(model.state_dict(), f"Tandem_model_fold_{fold+1}_MH.pth")
            else:
                if epoch >= 0:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch={epoch+1} for fold={fold+1}.")
                        break

        fold_results.append(best_fold_loss)

    avg_loss = sum(fold_results) / len(fold_results)
    print("\nK-Fold results:")
    for i, result in enumerate(fold_results):
        print(f" Fold {i+1}: Best Test Loss = {result:.6f}")
    print(f"Average Test Loss across folds = {avg_loss:.6f}")
