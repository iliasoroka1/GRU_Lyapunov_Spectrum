# GRU Lyapunov Spectrum

A recurrent neural network implementation for learning chaotic dynamical systems and computing their Lyapunov spectra. This project demonstrates the application of deep learning to the analysis of nonlinear dynamics using a double pendulum as a test case.

![Double Pendulum GRU](double_pendulum_GRU.gif)

## Overview

This repository provides tools for:
1. Training a custom Gated Recurrent Unit (GRU) network on chaotic time series data
2. Using the trained network as a surrogate dynamical system
3. Computing the full Lyapunov spectrum via Jacobian-based methods
4. Analyzing the stability and chaotic properties of the learned dynamics

## Lyapunov Exponents: Definition

The Lyapunov exponent quantifies the rate of separation of infinitesimally close trajectories in a dynamical system. For a dynamical system with state **x**(t), consider two nearby trajectories separated by an initial perturbation **δx**(0). The perturbation evolves as:

$$|\delta \mathbf{x}(t)| \approx |\delta \mathbf{x}(0)| e^{\lambda t}$$

The Lyapunov exponent λ is defined as:

$$\lambda = \lim_{t \to \infty} \lim_{|\delta \mathbf{x}(0)| \to 0} \frac{1}{t} \ln \frac{|\delta \mathbf{x}(t)|}{|\delta \mathbf{x}(0)|}$$

For an n-dimensional system, there exist n Lyapunov exponents (the **Lyapunov spectrum**), corresponding to the growth rates along different directions in phase space:

- **λ > 0**: Exponential divergence (sensitive dependence on initial conditions, chaos)
- **λ = 0**: Neutral stability (typically associated with the flow direction)
- **λ < 0**: Exponential convergence (dissipation)

A system is considered chaotic if at least one Lyapunov exponent is positive.

## Methodology

### State Representation

The double pendulum state is parameterized as **s** = [sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), ω₁, ω₂]ᵀ, where θᵢ and ωᵢ represent the angles and angular velocities of the two pendulum masses. This representation avoids discontinuities inherent in angular coordinates at the 2π boundary, improving training stability.

### Network Architecture

The implementation uses a custom GRU cell with the following update equations:

$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{x}_t, \mathbf{h}_{t-1}] + \mathbf{b}_z)$$

$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{x}_t, \mathbf{h}_{t-1}] + \mathbf{b}_r)$$

$$\hat{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \hat{\mathbf{h}}_t$$

The hidden state **h**_t ∈ ℝ⁶⁰⁰ is mapped to output space via a feedforward network with ReLU activation.

### Lyapunov Spectrum Computation

The algorithm follows the continuous QR decomposition method:

1. **Initialization**: Warm up the hidden state **h** using a trajectory segment from the training data
2. **Perturbation Setup**: Initialize M orthonormal perturbation vectors via QR decomposition of a random matrix
3. **Iteration**: For each time step:
   - Compute the Jacobian **J** = ∂**f**(**h**)/∂**h** using automatic differentiation (`torch.func.jacfwd`)
   - Propagate perturbations: **δh**' = **J** · **δh**
   - At regular intervals (every `norm_freq` steps), perform QR decomposition: **δh** = **QR**
   - Accumulate log growth rates: Σ ln|R_ii|
   - Advance the system: **h** ← **f**(**h**)
4. **Normalization**: Compute Lyapunov exponents as λᵢ = (Σ ln|R_ii|) / (total_time)

This approach treats the trained GRU as an autonomous discrete-time dynamical system and analyzes its stability properties in the learned hidden state space.

## example

```python
import torch
import numpy as np
from GRU import RNN

model = RNN(input_size=6, hidden_size=600, output_size=6)
model.load_state_dict(torch.load('model_pendulum.pth'))

data = np.load('s_sincos.npy')

# Compute Lyapunov spectrum
lyap_spectrum = model.lyapunov_exponents(
    initial_condition=data[10000:][::2],  # Initial trajectory segment
    n_steps=2500,                          # Number of integration steps
    dt=0.1,                                # Time step
    num_lyaps=500,                         # Number of exponents to compute
    warmup_steps=1,                        # Warmup period
    norm_freq=1,                           # QR decomposition frequency
    epsilon=1e-15,                         # Numerical stability parameter
    normalize_sincos=True                  # Enforce constraint sin²+cos²=1
)

print("Largest Lyapunov exponent:", lyap_spectrum[0])
```

### Training a New Model

```python
from torch.utils.data import DataLoader
from GRU import DatasetW, RNN, train_model
import torch.nn as nn

# Prepare dataset
seq_length = 50
train_dataset = DatasetW(data[10000:][::2], seq_length)
train_loader = DataLoader(train_dataset, batch_size=128, 
                         shuffle=True, drop_last=True)

# Initialize model
model = RNN(input_size=6, hidden_size=600, output_size=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configure training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 
                            lr=1e-4, 
                            weight_decay=1e-5,
                            betas=(0.9, 0.999))

# Train with normalization constraint
train_model(model, train_loader, criterion, optimizer,
           num_epochs=15, device=device, 
           normalize_sincos_values=True)

# Save checkpoint
torch.save(model.state_dict(), 'model_pendulum.pth')
```

### Generating Predictions

```python
from GRU import predict_future

# Generate autonomous predictions
predictions = predict_future(
    model=model,
    seed_data=data[:200],      # Initial condition
    future_steps=5000,         # Prediction horizon
    device=device,
    normalize_sincos_values=True
)
```

## Implementation Details

### Custom GRU Architecture

The GRU is implemented from scratch rather than using PyTorch's `nn.GRU` to enable:
- Direct access to internal weight matrices for Jacobian computation
- Custom normalization constraints during forward passes
- Explicit control over gradient flow for stability analysis

### Normalization Constraint

During training and autonomous prediction, the output is projected onto the constraint manifold:
$$
\sin(\theta_i) \leftarrow \frac{\sin(\theta_i)}{\sqrt{\sin^2(\theta_i) + \cos^2(\theta_i)}}
$$

$$
\cos(\theta_i) \leftarrow \frac{\cos(\theta_i)}{\sqrt{\sin^2(\theta_i) + \cos^2(\theta_i)}}
$$

This ensures that the learned dynamics respect the geometric structure of the angular state space.

