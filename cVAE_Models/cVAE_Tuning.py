#Tuning the hyperparamteres cVAE-AE
import optuna
from optuna.trial import TrialState
import os
import cv2
import numpy as np
import torch
import pandas as pd
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
import matplotlib.pyplot as plt
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
from Raw_Reflection_Structure_DataLoader import ReflStrDataLoader
from Models import Autoencoder

autoencoder_model_path = "/path to the autoencoder model/autoencoder_model_128_noise.pth"
ae_model = Autoencoder(latent_dim=128)
ae_model.load_state_dict(torch.load(autoencoder_model_path, map_location=DEVICE, weights_only=True))
ae_model.to(DEVICE)
ae_model.eval()

def get_latent_vector(ae_model, reflection_tensor):
    with torch.no_grad():
        _, c_latent = ae_model(reflection_tensor)
    return c_latent

data_loader = ReflStrDataLoader(
    start_index=0,
    end_index=14999, 
    folder_path="/path to data/",
    filename="Test_filenames_cvae_tuning.txt"
)

structures, reflections, file_names = data_loader.load_data()
reflections = np.expand_dims(reflections, axis=1)  

structures_tensor = torch.tensor(structures, dtype=torch.float32)
reflections_tensor = torch.tensor(reflections, dtype=torch.float32)

test_ratio = 0.01
trainval_idx, test_idx = train_test_split(np.arange(len(structures_tensor)), test_size=test_ratio, random_state=42)
X_trainval, X_test = structures_tensor[trainval_idx], structures_tensor[test_idx]
C_trainval, C_test = reflections_tensor[trainval_idx], reflections_tensor[test_idx]

def add_random_noise(x, noise_min=0.0, noise_max=0.1): 
    noise_levels = torch.rand(x.size(0), device=x.device) * (noise_max - noise_min) + noise_min 
    noise_levels = noise_levels.view(-1, *[1] * (x.ndim - 1))  
    noise = torch.randn_like(x) * noise_levels 
    x_noisy = x + noise
    return x_noisy

def train_cvae(model, train_loader, optimizer, device, noise_min=0.0, noise_max=0.1):
    model.train()
    total_loss, total_mse, total_mae, total_kl = 0, 0, 0, 0
    for x, c in train_loader:

        x = x.to(device) #structure
        c = c.to(device) #reflection as condition
        c_noise = add_random_noise(c, noise_min=noise_min, noise_max=noise_max)
        c_latent = get_latent_vector(ae_model, c_noise)       

        optimizer.zero_grad()
        x_recon, mu, logvar = model(x, c_latent)
        loss, mse_loss, mae_loss, kl = cvae_loss_function(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_mae += mae_loss.item()
        total_kl += kl.item()

    n_batches = len(train_loader)
    return (total_loss / n_batches,
            total_mse / n_batches,
            total_mae / n_batches,
            total_kl / n_batches)

def test_cvae(model, val_loader, device):
    model.eval()
    total_loss, total_mse, total_mae, total_kl = 0, 0, 0, 0
    
    with torch.no_grad():
        for x, c in val_loader:
            x = x.to(device)
            c = c.to(device)
            c_latent = get_latent_vector(ae_model, c)       
            
            x_recon, mu, logvar = model(x, c_latent)
            loss, mse_loss, mae_loss, kl = cvae_loss_function(x_recon, x, mu, logvar)
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_mae += mae_loss.item()
            total_kl += kl.item()
    
    n_batches = len(val_loader)
    return (total_loss / n_batches,
            total_mse / n_batches,
            total_mae / n_batches,
            total_kl / n_batches)


def cvae_loss_function(x_recon, x, mu, logvar):
    mse_loss = F.mse_loss(x_recon, x, reduction='mean')
    mae_loss = F.l1_loss(x_recon, x, reduction='mean')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return mse_loss + mae_loss + kl, mse_loss, mae_loss, kl

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}

class TunableCVAE(nn.Module):
    def __init__(self, trial, reflection_shape):
        super().__init__()
        self.condition_dim=128
        self.latent_dim = trial.suggest_categorical("latent_dim", [32, 64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        act_name = trial.suggest_categorical("activation", list(ACTIVATION_FUNCTIONS.keys()))
        act = ACTIVATION_FUNCTIONS[act_name]

        encoder_layers = []
        encoder_input_dim = self.condition_dim + 3
        num_encoder_layers = trial.suggest_int("encoder_layers", 3, 9)
        for i in range(num_encoder_layers):
            out_dim = trial.suggest_categorical(f"encoder_units_{i}", [32, 64, 128, 256])
            encoder_layers.append(nn.Linear(encoder_input_dim, out_dim))
            encoder_layers.append(act())
            encoder_layers.append(nn.LayerNorm(out_dim))
            encoder_layers.append(nn.Dropout(p=dropout))
            encoder_input_dim = out_dim
        self.encoder_fc = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_input_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_input_dim, self.latent_dim)

        decoder_input_dim = self.latent_dim + self.condition_dim
        decoder_layers = []
        num_decoder_layers = trial.suggest_int("decoder_layers", 3, 9)
        for i in range(num_decoder_layers):
            out_dim = trial.suggest_categorical(f"decoder_units_{i}", [32, 64, 128, 256])
            decoder_layers.append(nn.Linear(decoder_input_dim, out_dim))
            decoder_layers.append(act())
            decoder_layers.append(nn.LayerNorm(out_dim))
            decoder_layers.append(nn.Dropout(p=dropout))
            decoder_input_dim = out_dim
        decoder_layers.append(nn.Linear(decoder_input_dim, 3))  
        self.decoder_fc = nn.Sequential(*decoder_layers)

    def encode(self, x, c_latent):

        h = torch.cat([x, c_latent], dim=1)
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c_latent):
        return self.decoder_fc(torch.cat([z, c_latent], dim=1))

    def forward(self, x, c_latent):
        mu, logvar = self.encode(x, c_latent)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c_latent), mu, logvar


def objective(trial):
    model = TunableCVAE(trial, reflection_shape=(401, 90)).to(DEVICE)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1, eta_min=1e-7)

    X_train, X_val, C_train, C_val = train_test_split(X_trainval, C_trainval, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(X_train, C_train), batch_size=128, shuffle=True, num_workers=8)
    val_loader = DataLoader(TensorDataset(X_val, C_val), batch_size=128, shuffle=False, num_workers=8)

    num_epochs = 30
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss, *_ = train_cvae(model, train_loader, optimizer, DEVICE, noise_min=0.0, noise_max=0.1)
        val_loss, *_ = test_cvae(model, val_loader, DEVICE)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)  

    print("Best trial:")
    trial = study.best_trial
    print(f"Value:{trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"{key}:{value}")

    with open("best_hyperparameters.txt", "w") as f:
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
