# Instructions to Run Equilibrium Propagation Project

## Overview

We've modernized the Equilibrium Propagation project to use PyTorch. This guide helps you understand how to set up, train, and visualize the network.

## Requirements

- Python 3
- PyTorch, torchvision
- Other dependencies specified in `requirements.txt`

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop
   cd Towards-a-Biologically-Plausible-Backprop
   ```

2. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

### Training the Network

To train the network:

```bash
python train_model_pytorch.py net1
```

You can replace `net1` with `net2` or `net3` for different architectures.

### Visualizing the Network

Run the GUI to visualize the network:

```bash
python gui_pytorch.py net1
```

Make sure the network has been trained (i.e., `.save` file exists for the specified network name).

## Notes

- The training script downloads the MNIST dataset during the first run and converts it to tensors.
- Training and visualization should accommodate available hardware resources.

