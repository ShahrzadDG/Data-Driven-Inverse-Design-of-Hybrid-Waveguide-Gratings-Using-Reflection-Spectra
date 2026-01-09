# Inverse-Design-of-Hybrid-Waveguide-Grating-using-Reflection-Spectra
This repository contains the source codes and trained models used in our study "Data-Driven Inverse Design of Hybrid Waveguide Gratings using Reflection Spectra via Tandem Networks and Conditional VAEs". Paper: https://doi.org/10.3390/opt6040061

Problem: Predict structural parameters (Ag thickness, polymer thickness, period) from reflection spectra using deep learning.

Method: 

1. Data Generation & Simulation: Reflection spectra generation using Rigorous Coupled Wave Analysis (RCWA) simulation across the visible range and a complete span of incident angles. Dataset includes wide parameter space of hybrid waveguide gratings and their corresponding reflection spectra.
  
2. Autoencoder for dimension reduction: An autoencoder model is trained to compress 2D reflection spectra into a lower-dimensional latent representation. The AE model captures the essential spectral features while it reduces complexity. It is also the main denoising component of the framework. It learns stable latent space from the noisy reflection spectra. The latent vectors are later used as inputs to subsequent models, ensure that subsequent models are resilient to the noise and fluctuations typical of real-world measurements.

3. Forward model: A forward neural network learns the mapping from structural parameters to spectral latent representation (latent vectors from the AE model).
   
4. Inverse models: Tandem Network and conditional Variational Autoencoder (cVAE)

      1. Tandem Network: It combines a pre-trained forward model with an inverse network. The inverse model predicts the structural parameters from the latent representation of the reflection spectra. The predicted structural parameters are then used as input to the pre-trained forward model to reconstruct the latent representation of the reflection spectra. During training, the inverse model is optimized by minimizing the error between the predicted and ground-truth latent representations.
  
      2. Conditional Variational Autoencoder (cVAE): It uses spectral latent representations as the condition to generate probable structural parameters.


Results are provided in paper https://doi.org/10.3390/opt6040061

Tools: Rigorous coupled-wave analysis (RCWA) simulations for data collection, Python, PyTorch, Optuna 

The computations were carried out on the PLEIADES cluster at the University of Wuppertal, which was supported by the Deutsche Forschungsgemeinschaft (DFG, grant No. INST 218/78-1 FUGG) and the Bundesministerium für Bildung und Forschung (BMBF). [https://pleiades.uni-wuppertal.de]


To use the provided codes and models in this GitHub, users can run the provided codes, give paths to the trained models and their own reflection spectra to perform inverse design.

Notes for users:

The reflection spectra must be exactly the same shape, resolution and format as the sample reflection spectra provided in this GitHub repository. Please pay attention that there is headere in our sample data, either provide those headers or change the code accordingly to not lose any data points of your reflection spectra. Any mismatch in size or preprocessing may lead to incorrect predictions or error.
The trained models are valid only within the parameter ranges covered in the trained dataset. Extrapolation beyond these ranges may produce unreliable results.
