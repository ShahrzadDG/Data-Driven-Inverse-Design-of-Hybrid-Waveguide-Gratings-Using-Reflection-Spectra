import os
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import sys
sys.path.append('/path to codes/')
from AE_DataLoader import ReflectionDataLoader
from Models import Autoencoder

def add_random_noise(x, noise_min=0.0, noise_max=0.1): 
    noise_levels = torch.rand(x.size(0), device=x.device) * (noise_max - noise_min) + noise_min
    noise_levels = noise_levels.view(-1, *[1] * (x.ndim - 1))  
    noise = torch.randn_like(x) * noise_levels
    x_noisy = x + noise
    return x_noisy 

    
def train_autoencoder(model, train_loader, test_loader, num_epochs=20, learning_rate=1e-3, device='cpu', noise_min=0.0, noise_max=0.1 ,patience=10, decay_step=1000, decay_factor=0.1):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=decay_step, gamma=decay_factor)  

    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            noisy_inputs = add_random_noise(inputs, noise_min=0.0, noise_max=0.1)
            optimizer.zero_grad()
            outputs, _ = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
                
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, LR: {scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict()  
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
            
if __name__ == "__main__":
    start_index = 0
    end_index = 14999
    folder_path="path to data"
    test_size = 0.2
    rnd_numb = 42
    batch_size = 128
    latent_dim = 128
    num_epochs = 6000
    learning_rate = 0.00075
    patience = 200  
    decay_step = 1500  
    decay_factor = 0.1  
    noise_min = 0.0
    noise_max = 0.1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_loader = ReflectionDataLoader(start_index, end_index, folder_path, test_size, rnd_numb, batch_size)
    train_loader, test_loader = data_loader.load_data()

    model = Autoencoder(latent_dim=latent_dim)

    train_autoencoder(model, train_loader, test_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device, noise_min=noise_min, noise_max=noise_max, patience=patience, decay_step=decay_step, decay_factor=decay_factor)

    torch.save(model.state_dict(), "autoencoder_model_128_noise.pth")
