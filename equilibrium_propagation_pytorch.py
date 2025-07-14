import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os

class EquilibriumPropagationNetwork:
    """Modern PyTorch implementation of Equilibrium Propagation
    
    Based on the paper "Equilibrium Propagation: Bridging the Gap between 
    Energy-Based Models and Backpropagation" by Scellier & Bengio (2017)
    """
    
    def __init__(self, hidden_sizes=[500], device='cpu'):
        self.device = device
        self.hidden_sizes = hidden_sizes
        
        # Network architecture: 784 (MNIST) -> hidden_sizes -> 10
        layer_sizes = [784] + hidden_sizes + [10]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Glorot/Xavier initialization
            fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            W = torch.empty(fan_in, fan_out).uniform_(-bound, bound).to(device)
            W.requires_grad_(True)
            self.weights.append(W)
            
            # Initialize biases to zero
            b = torch.zeros(layer_sizes[i+1]).to(device)
            b.requires_grad_(True)
            self.biases.append(b)
    
    def rho(self, s):
        """Hard sigmoid activation function: rho(s) = max(0, min(1, s))"""
        return torch.clamp(s, 0.0, 1.0)
    
    def energy(self, layers):
        """Compute energy function E for the current state"""
        # Squared norm term: sum(rho(layer)^2) / 2
        squared_norm = sum([(self.rho(layer) ** 2).sum(dim=1) for layer in layers]) / 2.0
        
        # Linear terms: -sum(rho(layer) * bias) - note: skip input layer for biases
        linear_terms = -sum([torch.sum(self.rho(layer) * bias, dim=1) 
                           for layer, bias in zip(layers[1:], self.biases)])
        
        # Quadratic terms: -sum(rho(pre) @ W @ rho(post))
        quadratic_terms = -sum([torch.sum(self.rho(pre) @ W * self.rho(post), dim=1)
                               for pre, W, post in zip(layers[:-1], self.weights, layers[1:])])
        
        return squared_norm + linear_terms + quadratic_terms
    
    def cost(self, output_layer, target):
        """Compute cost function C (mean squared error)"""
        target_one_hot = F.one_hot(target, num_classes=10).float()
        return ((output_layer - target_one_hot) ** 2).sum(dim=1)
    
    def total_energy(self, layers, target, beta):
        """Compute total energy F = E + beta * C"""
        return self.energy(layers) + beta * self.cost(layers[-1], target)
    
    def free_phase(self, x, n_iterations=20, epsilon=0.5):
        """Run free phase dynamics to find equilibrium"""
        batch_size = x.shape[0]
        
        # Initialize layers
        layers = [x]  # Input layer (clamped)
        for size in self.hidden_sizes + [10]:
            layer = torch.zeros(batch_size, size, requires_grad=True).to(self.device)
            layers.append(layer)
        
        # Run dynamics
        for _ in range(n_iterations):
            # Compute energy gradient
            energy_val = self.energy(layers).sum()
            grads = torch.autograd.grad(energy_val, layers[1:], create_graph=True)
            
            # Update layers (except input which is clamped)
            new_layers = [layers[0]]  # Keep input layer
            for i, grad in enumerate(grads):
                new_layer = self.rho(layers[i+1] - epsilon * grad)
                new_layer.requires_grad_(True)
                new_layers.append(new_layer)
            layers = new_layers
        
        return layers
    
    def weakly_clamped_phase(self, x, target, n_iterations=4, epsilon=0.5, beta=0.5):
        """Run weakly clamped phase dynamics"""
        batch_size = x.shape[0]
        
        # Initialize layers
        layers = [x]  # Input layer (clamped)
        for size in self.hidden_sizes + [10]:
            layer = torch.zeros(batch_size, size, requires_grad=True).to(self.device)
            layers.append(layer)
        
        # Run dynamics
        for _ in range(n_iterations):
            # Compute total energy gradient
            total_energy_val = self.total_energy(layers, target, beta).sum()
            grads = torch.autograd.grad(total_energy_val, layers[1:], create_graph=True)
            
            # Update layers (except input which is clamped)
            new_layers = [layers[0]]  # Keep input layer
            for i, grad in enumerate(grads):
                new_layer = self.rho(layers[i+1] - epsilon * grad)
                new_layer.requires_grad_(True)
                new_layers.append(new_layer)
            layers = new_layers
        
        return layers
    
    def compute_gradients(self, layers_free, layers_clamped, beta):
        """Compute parameter gradients using equilibrium propagation"""
        # Energy at free equilibrium
        energy_free = self.energy(layers_free).mean()
        
        # Energy at weakly clamped equilibrium  
        energy_clamped = self.energy(layers_clamped).mean()
        
        # Gradient of energy difference w.r.t. parameters
        energy_diff = (energy_clamped - energy_free) / beta
        
        weight_grads = torch.autograd.grad(energy_diff, self.weights, retain_graph=True)
        bias_grads = torch.autograd.grad(energy_diff, self.biases[1:])  # Don't update input bias
        
        return weight_grads, bias_grads
    
    def update_parameters(self, weight_grads, bias_grads, alphas):
        """Update network parameters"""
        with torch.no_grad():
            for i, (W, grad) in enumerate(zip(self.weights, weight_grads)):
                W -= alphas[i] * grad
            
            for i, (b, grad) in enumerate(zip(self.biases[1:], bias_grads)):
                b -= alphas[i] * grad
    
    def predict(self, x):
        """Make predictions using free phase"""
        layers = self.free_phase(x)
        return torch.argmax(layers[-1], dim=1)
    
    def measure(self, x, target):
        """Measure energy, cost, and error rate"""
        layers = self.free_phase(x)
        
        energy_val = self.energy(layers).mean()
        cost_val = self.cost(layers[-1], target).mean()
        
        predictions = torch.argmax(layers[-1], dim=1)
        error_rate = (predictions != target).float().mean()
        
        return energy_val.item(), cost_val.item(), error_rate.item()

def load_mnist_data(batch_size=20):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_network(hidden_sizes=[500], n_epochs=25, batch_size=20, 
                 n_it_neg=20, n_it_pos=4, epsilon=0.5, beta=0.5, 
                 alphas=[0.1, 0.05], device='cpu'):
    """Train equilibrium propagation network"""
    
    print(f"Architecture: 784-{'-'.join(map(str, hidden_sizes))}-10")
    print(f"Epochs: {n_epochs}, Batch size: {batch_size}")
    print(f"Free phase iterations: {n_it_neg}, Clamped phase iterations: {n_it_pos}")
    print(f"Learning rate: {epsilon}, Beta: {beta}")
    print(f"Alphas: {alphas}\n")
    
    # Initialize network
    net = EquilibriumPropagationNetwork(hidden_sizes, device)
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # Training curves
    training_errors = []
    validation_errors = []
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        
        # Training phase
        train_energy_sum = 0.0
        train_cost_sum = 0.0
        train_error_sum = 0.0
        num_train_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Free phase
            layers_free = net.free_phase(data, n_it_neg, epsilon)
            
            # Measure at free equilibrium
            energy, cost, error = net.measure(data, target)
            train_energy_sum += energy
            train_cost_sum += cost
            train_error_sum += error
            num_train_batches += 1
            
            if batch_idx % 100 == 0:
                avg_error = train_error_sum / num_train_batches * 100
                print(f"  Batch {batch_idx}: Error = {avg_error:.2f}%")
            
            # Weakly clamped phase
            sign = 2 * np.random.randint(0, 2) - 1  # Random sign
            beta_signed = sign * beta
            
            layers_clamped = net.weakly_clamped_phase(data, target, n_it_pos, epsilon, beta_signed)
            
            # Compute and apply gradients
            weight_grads, bias_grads = net.compute_gradients(layers_free, layers_clamped, beta_signed)
            net.update_parameters(weight_grads, bias_grads, alphas)
        
        avg_train_error = train_error_sum / num_train_batches * 100
        training_errors.append(avg_train_error)
        print(f"  Training error: {avg_train_error:.2f}%")
        
        # Validation phase
        val_energy_sum = 0.0
        val_cost_sum = 0.0
        val_error_sum = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                energy, cost, error = net.measure(data, target)
                val_energy_sum += energy
                val_cost_sum += cost
                val_error_sum += error
                num_val_batches += 1
        
        avg_val_error = val_error_sum / num_val_batches * 100
        validation_errors.append(avg_val_error)
        
        duration = (time.time() - start_time) / 60.0
        print(f"  Validation error: {avg_val_error:.2f}%")
        print(f"  Duration: {duration:.1f} min\n")
    
    return net, training_errors, validation_errors

def plot_training_curves(training_errors, validation_errors):
    """Plot training and validation error curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(training_errors, label='Training Error')
    plt.plot(validation_errors, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.title('Equilibrium Propagation Training Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Train a network with 1 hidden layer (similar to original net1)
    net, train_errors, val_errors = train_network(
        hidden_sizes=[500],
        n_epochs=5,  # Reduced for demo
        batch_size=20,
        n_it_neg=20,
        n_it_pos=4,
        epsilon=0.5,
        beta=0.5,
        alphas=[0.1, 0.05],
        device=device
    )
    
    # Plot results
    plot_training_curves(train_errors, val_errors)
    
    print("Training completed!")
    print(f"Final training error: {train_errors[-1]:.2f}%")
    print(f"Final validation error: {val_errors[-1]:.2f}%")
