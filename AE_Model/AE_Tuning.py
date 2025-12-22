#optuna tuning 
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append('/path to codes/')
from AE_DataLoader import ReflectionDataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def add_random_noise(x, noise_min=0.0, noise_max=0.1): #noise addded to the reflection data
    noise_levels = torch.rand(x.size(0), device=x.device) * (noise_max - noise_min) + noise_min 
    noise_levels = noise_levels.view(-1, *[1] * (x.ndim - 1))  
    noise = torch.randn_like(x) * noise_levels 
    x_noisy = x + noise
    return x_noisy 
    

class AE_tunung_Optuna(nn.Module):
    def __init__(self, trial, input_shape=(1, 401, 90)):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = trial.suggest_categorical("latent_dim", [32, 64, 128, 256])
        # print(self.latent_dim)
        n_layers = trial.suggest_int("n_layers", 3, 7)
        # print(n_layers)
        in_channels = 1
        modules_enc = []
        decoder_modules = []
        conv_config = []
        conv_layers = []

        for i in range(n_layers):
            out_channels = trial.suggest_categorical(f"conv_{i}_out_channels", [16, 32, 64, 128])
            modules_enc.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            modules_enc.append(nn.ReLU(True))
            conv_config.append((in_channels, out_channels))
            conv_layers.append({'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 1})
            # print(conv_layers)
            in_channels = out_channels
            # print(in_channels)
            # print(out_channels)
        # print(input_shape[1], input_shape[2])
        h, w = self.compute_output_size(input_shape[1], input_shape[2], conv_layers)
        # print(h ,w)
        # self.H3, self.W3, self.final_channels = h, w, in_channels
        modules_enc.append(nn.Flatten())
        modules_enc.append(nn.Linear(in_channels * h * w, self.latent_dim))
        modules_enc.append(nn.ReLU(True))
        modules_enc.append(nn.LayerNorm(self.latent_dim))  
        self.encoder = nn.Sequential(*modules_enc)

        decoder_modules = [nn.Linear(self.latent_dim, in_channels * h * w),nn.ReLU(True),nn.Unflatten(1, (in_channels, h, w))]

        for i, (enc_in, enc_out) in enumerate(reversed(conv_config)):
            # print(enc_out, enc_in)
            is_last = (i == len(conv_config) - 1)
            # print(i , is_last)

            decoder_modules.append(nn.ConvTranspose2d(enc_out, enc_in, kernel_size=3, stride=2, padding=1, output_padding=1))
        #     decoder_modules.append(nn.ReLU())

        # decoder_modules.append(nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1))  
        # decoder_modules.append(nn.Sigmoid())
    
            if is_last:
                decoder_modules.append(nn.Sigmoid())
            else:
                decoder_modules.append(nn.ReLU(True))
        
        self.decoder = nn.Sequential(*decoder_modules)

    def compute_output_size(self, H, W, conv_layers):
        for layer in conv_layers:
            kernel_size = layer.get('kernel_size', 3)
            stride = layer.get('stride', 2)
            padding = layer.get('padding', 1)
            dilation = layer.get('dilation', 1)
            H = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            W = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        return H, W
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        # print(reconstructed.size(2), reconstructed.size(3))
        diff_h = reconstructed.size(2) - x.size(2)  
        diff_w = reconstructed.size(3) - x.size(3)  

        if diff_h < 0 or diff_w < 0:
            raise ValueError("Error (small reconstructed image)")

        crop_h_start = diff_h // 2
        crop_h_end = crop_h_start + x.size(2)
        crop_w_start = diff_w // 2
        crop_w_end = crop_w_start + x.size(3)

        reconstructed = reconstructed[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
        # print(reconstructed.shape)
        return reconstructed, latent

def objective(trial):
    model = AE_tunung_Optuna(trial).to(device)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    if optimizer_name == "SGD":
        lr = trial.suggest_loguniform("lr", 1e-3, 1e-1)
    else:
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    # lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_epochs = 30  
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            noisy_inputs = add_random_noise(inputs, noise_min=0.0, noise_max=0.1)
            # print(inputs.shape)
       
            optimizer.zero_grad()
            outputs, _ = model(inputs)

            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(total_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                # print(inputs.shape)
                outputs, _ = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)
        # print(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    return best_val_loss


if __name__ == "__main__":
    start_index = 0
    end_index = 14999
    folder_path="path to data"
    test_size = 0.2
    rnd_numb = 42
    batch_size = 128
    
    data_loader = ReflectionDataLoader(start_index, end_index, folder_path, test_size, rnd_numb, batch_size)
    train_loader, test_loader = data_loader.load_data()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=300)
    print("Best trial:")
    trial = study.best_trial
    
    print(f"Loss: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
    best_model = AE_tunung_Optuna(trial).to(device)
