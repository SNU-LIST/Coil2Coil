# CoiltoCoil
* The code is for denoising of MR images using deep learning (C2C)
* last update : 2022. 07. 04

# Overview

![figure 1](/fig1.png)

# Requirements 
* Python 3.7

* pytorch=1.9.0

* NVIDIA GPU 


# Usage
### Simulation 

* Monte-Carlo diffusion simulation code to generate diffusion-weighted signals for training.

### Training

* The source code for training DIFFnet. Simulated data from Monte-Carlo diffusion simulation has to be required.

### Evaluation

* The source code for evaluation of the trained networks.
* In-vivo data and simulated data can be evaluated both.
* Networks generate diffusion model parameters.

