# Instructions for Running and Extending the Equilibrium Propagation Project

## Overview

This project implements Equilibrium Propagation using Theano. It demonstrates how biologically plausible learning can be applied to neural networks and provides a foundation for further experiments and adaptations.

## Requirements

- Python 3
- [Theano](https://github.com/Theano/Theano)
- Additional Python libraries: NumPy, Tkinter, PIL (Pillow)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop.git
   cd Towards-a-Biologically-Plausible-Backprop
   ```

2. **Install Required Packages:**
   ```bash
   pip install theano numpy pillow
   ```
   Ensure you have `Tkinter` installed, which is included with standard Python installs on MacOS. If issues arise, consult system package managers.

3. **Configure Theano:**
  - For better performance, configure Theano with desired compute capability.
  - Example configuration command:
    ```bash
    THEANO_FLAGS="floatX=float32,device=gpu"
    ```

4. **Download MNIST Dataset:** The dataset will be downloaded automatically when you run the code if it's not 

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

## Troubleshooting and Common Issues

- If you face issues with Theano configurations, refer to [Theano's official documentation](https://theano.readthedocs.io/en/latest/).
- Ensure all dependencies are correctly installed. Use virtual environments to isolate dependencies.

## Extending to Other Problems

To apply this code to different problems:

1. **Adapt `external_world.py` for new datasets:** Adjust the data loading and preprocessing steps.
2. **Adjust network input/output layers** in `model.py` to match the shape of the new data.
3. **Experiment with hyperparameters** to optimize performance on the new task.
4. **Integrate with quantum annealing**: Explore potential alignment with quantum computing methods where applicable.

## Suggested Experimentation Path

1. **Start with the default settings** in `train_model.py` to get familiar with the training process.
2. **Experiment with multiple configurations** by changing hyperparameters.
3. **Integrate new data** to explore different applications of equilibrium propagation.
4. **Delve into advanced models:** Once familiar, experiment with integrating into hybrid models like quantum-classical systems.

## Further Exploration

For those interested in cutting-edge applications:

1. **Quantum Computing:** The potential synergy between equilibrium propagation and quantum annealing may offer exciting new research avenues.
2. **Other Neuro-inspired Models:** Extend and test with spiking neural networks if applicable. Keep abreast of the latest research for innovative applications.

