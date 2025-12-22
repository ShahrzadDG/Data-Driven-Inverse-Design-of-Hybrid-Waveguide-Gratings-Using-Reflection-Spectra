import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import sys
sys.path.append('/path to the codes/')
from Reflection_Structure_DataLoader import ReflStrDataLoader

def objective(trial):
    num_layers = trial.suggest_int('num_layers', 3, 9) 
    n_units_per_layer = [trial.suggest_int(f'n_units_l{i}', 64, 512, step=32) for i in range(num_layers)]

    activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
    if activation_name == 'ReLU':
        activation = nn.ReLU()
    elif activation_name == 'LeakyReLU':
        negative_slope = trial.suggest_float('leaky_relu_negative_slope', 1e-3, 0.3)
        activation = nn.LeakyReLU(negative_slope=negative_slope)
        # activation = nn.LeakyReLU()
    else:
        activation = nn.Tanh()
    
    layers = []
    input_dim = 3
    output_dim = 128
    layers.append(nn.Linear(input_dim, n_units_per_layer[0]))
    layers.append(activation)

    for i in range(1, num_layers):
        layers.append(nn.Linear(n_units_per_layer[i-1], n_units_per_layer[i]))
        layers.append(activation)

    layers.append(nn.Linear(n_units_per_layer[-1], output_dim))
    
    model = nn.Sequential(*layers)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True))
    scheduler = StepLR(optimizer, step_size=1500, gamma=trial.suggest_float('decay_factor', 0.1, 0.9))
    
    num_epochs = 30
    patience = 500
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
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(test_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            break

    return best_val_loss

if __name__ == "__main__":
    start_index = 0
    end_index = 14999
    folder_path="path to data"
    test_size = 0.2
    rnd_numb = 42
    batch_size = 128

    data_loader = ReflStrDataLoader(start_index, end_index, folder_path, test_size, rnd_numb, batch_size, inverse=False, filename="Test_filenames-forward_tuning.txt")
    train_loader, test_loader = data_loader.load_data()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200)  

    print("Best hyperparameters: ", study.best_params)

    torch.save(study.best_trial, "Best_optuna_trial_StrtoLatent.pth")
