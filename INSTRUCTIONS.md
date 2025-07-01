# Instructions for Running and Extending the Equilibrium Propagation Project

## Overview

This project implements Equilibrium Propagation using Theano. It demonstrates how biologically plausible learning can be applied to neural networks and provides a foundation for further experiments and adaptations.

## Requirements

- Python 3
- [Theano](https://github.com/Theano/Theano)
- Additional Python libraries: NumPy, Tkinter, PIL (Pillow)

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop.git
   cd Towards-a-Biologically-Plausible-Backprop
   ```

2. **Install Required Packages:** Ensure that Theano, NumPy, Tkinter, and PIL are installed.

3. **Download MNIST Dataset:** The dataset will be downloaded automatically when you run the code if it's not already present.

## Running the Code

- To train a Hopfield network with one hidden layer using Equilibrium Propagation:
  ```bash
  THEANO_FLAGS="floatX=float32, gcc.cxxflags='-march=core2'" python train_model.py
  ```
- Once the network is trained, visualize it using the GUI:
  ```bash
  THEANO_FLAGS="floatX=float32, gcc.cxxflags='-march=core2'" python gui.py net1
  ```

## Understanding the Code

- **train_model.py:** Contains functions to train the neural network. You can modify hyperparameters for different network configurations.
- **model.py:** Defines the network structure and learning phases. Here, you control the learning dynamics.
- **external_world.py:** Loads and preprocesses the dataset.
- **gui.py:** Provides a graphical interface for visualizing the trained network.

## Extending to Other Problems

To apply this code to different problems:

1. **Adapt `external_world.py` for new datasets:** Adjust the data loading and preprocessing steps.
2. **Adjust network input/output layers** in `model.py` to match the shape of the new data.
3. **Experiment with hyperparameters** to optimize performance on the new task.

## Suggested Experimentation Path

1. **Start with the default settings** in `train_model.py` to get familiar with the training process.
2. **Experiment with multiple configurations** by changing hyperparameters.
3. **Integrate new data** to explore different applications of equilibrium propagation.

