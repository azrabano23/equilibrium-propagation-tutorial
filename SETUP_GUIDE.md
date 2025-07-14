# Complete Setup Guide for Equilibrium Propagation Project

## Overview

This guide provides step-by-step instructions to set up and run the Equilibrium Propagation project, which has been modernized from the original Theano implementation to PyTorch.

## What is Equilibrium Propagation?

Equilibrium Propagation is a biologically plausible learning algorithm for energy-based models. It uses two phases:
- **Free Phase**: The network relaxes to equilibrium without target clamping
- **Weakly Clamped Phase**: The network relaxes with weak target signal

The learning rule is based on the energy difference between these two phases.

## File Structure

```
Towards-a-Biologically-Plausible-Backprop/
├── README.md                      # Original repository README
├── requirements.txt               # Python dependencies
├── SETUP_GUIDE.md                # This file
├── INSTRUCTIONS.md               # Quick instructions
├── external_world_pytorch.py     # Data loading (PyTorch)
├── model_pytorch.py             # Network model (PyTorch)
├── train_model_pytorch.py       # Training script (PyTorch)
├── gui_pytorch.py               # Visualization GUI (PyTorch)
├── external_world.py            # Original data loading (Theano)
├── model.py                     # Original network model (Theano)
├── train_model.py               # Original training script (Theano)
├── gui.py                       # Original GUI (Theano)
└── data/                        # MNIST dataset (downloaded automatically)
```

## System Requirements

- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.8 or higher
- **Memory**: At least 4GB RAM
- **Storage**: At least 1GB free space

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop
cd Towards-a-Biologically-Plausible-Backprop
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torchvision; print('Torchvision version:', torchvision.__version__)"
```

## Running the Project

### Training a Network

The project includes three pre-configured networks:

1. **net1**: Single hidden layer (500 units) - Fastest to train
2. **net2**: Two hidden layers (500 units each) - Moderate training time
3. **net3**: Three hidden layers (500 units each) - Longest training time

To train a network:

```bash
python train_model_pytorch.py net1
```

**Training Parameters:**
- **net1**: 25 epochs, ~1 hour on CPU
- **net2**: 60 epochs, ~3 hours on CPU  
- **net3**: 500 epochs, ~24 hours on CPU

### Visualizing the Network

After training, launch the GUI:

```bash
python gui_pytorch.py net1
```

**GUI Features:**
- Real-time visualization of network layers
- Interactive image selection
- Energy, cost, and prediction display
- Navigation controls (Next, Previous, Random)

### Understanding the Output

During training, you'll see:
- **Energy (E)**: Network energy at equilibrium
- **Cost (C)**: Mean squared error with targets
- **Error**: Classification error percentage
- **Weight changes**: Relative parameter updates

## Advanced Usage

### Custom Network Architectures

You can modify the network architecture by editing the hyperparameters in `train_model_pytorch.py`:

```python
custom_config = {
    "hidden_sizes": [300, 200],  # Two hidden layers
    "n_epochs": 50,
    "batch_size": 20,
    "n_it_neg": 30,      # Free phase iterations
    "n_it_pos": 6,       # Weakly clamped phase iterations
    "epsilon": 0.5,      # Learning rate
    "beta": 0.8,         # Perturbation strength
    "alphas": [0.1, 0.05, 0.01]  # Layer-wise learning rates
}
```

### Using GPU Acceleration

The code automatically detects and uses GPU if available:

```bash
# Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Saving and Loading Models

Networks are automatically saved during training as `.save` files. To load a pre-trained network:

```python
from model_pytorch import Network

# Load existing network
net = Network("net1", device='cpu')
```

## Google Colab Support

To run in Google Colab:

1. **Upload the files** to your Colab environment
2. **Install dependencies**:
   ```python
   !pip install torch torchvision tqdm Pillow
   ```
3. **Run training**:
   ```python
   !python train_model_pytorch.py net1
   ```

Note: GUI visualization won't work in Colab, but you can use the training script and examine the results programmatically.

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size or use smaller network
2. **Slow Training**: Use GPU or reduce number of iterations
3. **GUI Won't Start**: Check that you have tkinter installed and X11 forwarding enabled (Linux/macOS)

### Performance Tips

- Use GPU for faster training
- Reduce `n_it_neg` and `n_it_pos` for faster convergence
- Start with `net1` before trying larger networks

## Understanding the Algorithm

### Key Concepts

1. **Energy Function**: E(s) = ½∑ρ(sᵢ)² - ∑bᵢρ(sᵢ) - ∑Wᵢⱼρ(sᵢ)ρ(sⱼ)
2. **Hard Sigmoid**: ρ(s) = clip(s, 0, 1)
3. **Learning Rule**: Δθ ∝ (E_clamped - E_free) / β

### Network Dynamics

- **Free Phase**: ds/dt = -∂E/∂s
- **Weakly Clamped Phase**: ds/dt = -∂(E + βC)/∂s
- **Parameter Updates**: Based on energy difference

## Extensions and Applications

The repository also includes documentation for advanced applications:
- Quantum annealing integration
- Neuromorphic computing
- Continual learning
- Multi-agent systems

See `ADVANCED_APPLICATIONS.md` for more details.

## References

- Original Paper: [Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179)
- Original Repository: [https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop](https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure you have sufficient system resources
4. Try with a simpler network configuration first
