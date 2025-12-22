import os
import cv2
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        
        self.reflection_height = 401
        self.reflection_width = 90
        
        self.H3, self.W3 = self.compute_output_size()

        #encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * self.H3 * self.W3, latent_dim),
            nn.ReLU(True),
            nn.LayerNorm(latent_dim)
        )

        #decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.H3 * self.W3),
            nn.ReLU(True),
            nn.Unflatten(1, (128, self.H3, self.W3)),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  
        )
        
    def compute_output_size(self):
        H = self.reflection_height
        W = self.reflection_width
        conv_layers = [
            {'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'kernel_size': 3, 'stride': 2, 'padding': 1}
        ]
        for layer in conv_layers:
            kernel_size = layer['kernel_size']
            stride = layer['stride']
            padding = layer['padding']
            dilation = 1
            H = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            W = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        return H, W
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        
        diff_h = reconstructed.size(2) - x.size(2)  
        diff_w = reconstructed.size(3) - x.size(3)  

        if diff_h < 0 or diff_w < 0:
            raise ValueError("Reconstructed < input")

        crop_h_start = diff_h // 2
        crop_h_end = crop_h_start + x.size(2)
        crop_w_start = diff_w // 2
        crop_w_end = crop_w_start + x.size(3)

        reconstructed = reconstructed[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]

        return reconstructed, latent

#Forward architect
# Best hyperparameters:  {'num_layers': 6, 'n_units_l0': 416, 'n_units_l1': 416, 'n_units_l2': 96, 'n_units_l3': 256, 
# 'n_units_l4': 288, 'n_units_l5': 416, 'activation': 'Tanh', 'lr': 0.0016734102529168756, 'decay_factor': 0.8347596371170066}
class StructureToLatent(nn.Module):
    def __init__(self, input_dim=3, output_dim=128):
        super(StructureToLatent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 416)
        self.fc2 = nn.Linear(416, 416)
        self.fc3 = nn.Linear(416, 96)
        self.fc4 = nn.Linear(96, 256)
        self.fc5 = nn.Linear(256, 288)
        self.fc6 = nn.Linear(288, 416)
        self.fc7 = nn.Linear(416, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.activation(self.fc6(x))
        x = self.fc7(x)  
        return x

#Tandem Training
# Best trial:
#   Value: 0.023065245438677568
#   Params: 
#     n_layers: 4
#     n_units_l0: 352
#     n_units_l1: 512
#     n_units_l2: 192
#     n_units_l3: 288
#     dropout_rate: 0.1183144265309136
#     activation: ReLU
#     learning_rate: 0.00040305563500179175
#     decay_step: 2196
#     decay_factor: 0.5284655850306843
class LatenttoStructureMH(nn.Module):
    def __init__(self, input_dim=128, output_dim=3):
        super(LatenttoStructureMH, self).__init__()
        self.dropout_rate = 0.1
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 352),
            nn.LayerNorm(352),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(352, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        self.head_silver = nn.Sequential(
            nn.Linear(192, 288),
            nn.ReLU(),
            nn.LayerNorm(288),
            nn.ReLU(),
            nn.Linear(288, 1)
        )
        self.head_ormocore = nn.Sequential(
            nn.Linear(192, 288),       
            nn.ReLU(),
            nn.LayerNorm(288),
            nn.ReLU(),
            nn.Linear(288, 1)
        )
        self.head_period = nn.Sequential(
            nn.Linear(192, 288),
            nn.ReLU(),
            nn.LayerNorm(288),
            nn.ReLU(),
            nn.Linear(288, 1)
        )

    def forward(self, x):
        f = self.shared(x)
        y_ag   = self.head_silver(f)
        y_ormo = self.head_ormocore(f)
        y_per  = self.head_period(f)
        return torch.cat([y_ag, y_ormo, y_per], dim=1)
    
class TandemNetwork(nn.Module):
    def __init__(self, forward_model, inverse_model):
        super(TandemNetwork, self).__init__()

        self.forward_model = forward_model
        self.inverse_model = inverse_model

    def forward(self, y):
        # x:structure, y:reflection
        pred_structure = self.inverse_model(y)
        latent_pred = self.forward_model(pred_structure)
        return latent_pred

    def pred(self, y):
        pred_structure = self.inverse_model(y)
        return pred_structure


# cVAE architect
# Best trial:
#   Value: 0.015586304944008589
#   Params:
#     latent_dim: 32
#     dropout: 0.00036045162627821193
#     activation: relu
#     encoder_layers: 7
#     encoder_units_0: 128
#     encoder_units_1: 256
#     encoder_units_2: 256
#     encoder_units_3: 128
#     encoder_units_4: 64
#     encoder_units_5: 64
#     encoder_units_6: 32
#     decoder_layers: 4
#     decoder_units_0: 64
#     decoder_units_1: 256
#     decoder_units_2: 256
#     decoder_units_3: 64
#     lr: 0.0007485453508139011
#cVAE_Training has dropout = 0.0004 
#cVAE_Training_2 has dropout = 0.1
class cVAEtoStructureMH(nn.Module):
    def __init__(self, latent_dim=32, encoder_dropout=0.1, decoder_dropout=0.1, head_hidden=64):
        super(cVAEtoStructureMH, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = 128

        encoder_in_dim = self.condition_dim + 3
        self.encoder_fc = nn.Sequential(
            nn.Linear(encoder_in_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(p=encoder_dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(p=encoder_dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(p=encoder_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(p=encoder_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(p=encoder_dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(p=encoder_dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        self.fc_mu = nn.Linear(32, self.latent_dim)
        self.fc_logvar = nn.Linear(32, self.latent_dim)

        decoder_in_dim = self.latent_dim + self.condition_dim
        self.decoder_shared = nn.Sequential(
            nn.Linear(decoder_in_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(p=decoder_dropout),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(p=decoder_dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(p=decoder_dropout)
        )

        self.head_silver = nn.Sequential(
            nn.Linear(256, head_hidden),
            nn.ReLU(),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, 1)
        )
        self.head_ormocore = nn.Sequential(
            nn.Linear(256, head_hidden),
            nn.ReLU(),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, 1)
        )
        self.head_period = nn.Sequential(
            nn.Linear(256, head_hidden),
            nn.ReLU(),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, 1)
        )

    
    def encode(self, x, c_latent):
    
        combined = torch.cat([x, c_latent], dim=1)      
        h = self.encoder_fc(combined)                

        mu = self.fc_mu(h)                            
        logvar = self.fc_logvar(h)                    
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c_latent):
        combined = torch.cat([z, c_latent], dim=1)  
        feat = self.decoder_shared(combined)
        y_silver   = self.head_silver(feat)    
        y_ormocore = self.head_ormocore(feat)  
        y_period   = self.head_period(feat)    
        x_recon = torch.cat([y_silver, y_ormocore, y_period], dim=1)
        return x_recon

    def forward(self, x, c_latent):
        mu, logvar = self.encode(x, c_latent)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c_latent)
        return x_recon, mu, logvar

    