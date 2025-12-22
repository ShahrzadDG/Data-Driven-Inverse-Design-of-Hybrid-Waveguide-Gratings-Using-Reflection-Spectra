# Inverse-Design-of-Hybrid-Waveguide-Grating-using-Reflection-Spectra
This repository contains the source codes and trained models used in our study "Data-Driven Inverse Design of Hybrid Waveguide Gratings using Reflection Spectra via Tandem Networks and Conditional VAEs". Paper: https://doi.org/10.3390/opt6040061

Problem: Predict structural parameters (Ag thickness, polymer thickness, period) from reflection spectra using deep learning.
Method: 
1. Autoencoder for spectral compression
2. Pretrained Forward model to be used later in inverse design 
3. Conditional VAE and Tandem Network for inverse design
Results are provided in paper https://doi.org/10.3390/opt6040061

Tools: Rigorous coupled-wave analysis (RCWA) simulations for data collection, Python, PyTorch, Optuna 

Using the provided code, users can run the trained models on their own reflection spectra to perform inverse design.

Notes for users:

The reflection spectra must be exactly the same shape, resolution and format as the sample reflection spectra provided in this GitHub repository. Please pay attention that there is headere in our sample data, either provide those headers or change the code accordingly to not lose any data points of your reflection spectra. Any mismatch in size or preprocessing may lead to incorrect predictions or error.
The trained models are valid only within the parameter ranges covered in the trained dataset. Extrapolation beyond these ranges may produce unreliable results.
