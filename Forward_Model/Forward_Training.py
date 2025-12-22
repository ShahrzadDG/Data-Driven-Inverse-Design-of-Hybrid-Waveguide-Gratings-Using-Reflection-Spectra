import os
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
#GPU Usage
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

import sys
sys.path.append('/path to codes/')
from Reflection_Structure_DataLoader import ReflStrDataLoader
from Models import Autoencoder, StructureToLatent

def train_ForwardModel(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-3, device='cpu', patience=10, decay_step=1000, decay_factor=0.1):
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
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
                
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
    start_index=0 
    end_index=14999
    folder_path = "/path to data/"
    test_size = 0.2
    rnd_numb = 42
    batch_size = 128
    latent_dim = 128
    num_epochs = 10000
    learning_rate = 0.00167
    patience = 400
    decay_step = 500  
    decay_factor = 0.8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_loader = ReflStrDataLoader(start_index, end_index, folder_path, test_size, rnd_numb, batch_size, inverse=False, filename="Test_filenames_forward_training.txt")
    train_loader, test_loader = data_loader.load_data()

    model = StructureToLatent(input_dim=3, output_dim=128)

    train_ForwardModel(model, train_loader, test_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device, patience=patience, decay_step=decay_step, decay_factor=decay_factor)

    torch.save(model.state_dict(), "Forward_model_Strtolatent_optuna_NoisyAE.pth")
