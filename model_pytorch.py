import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
from external_world_pytorch import External_World

def rho(s):
    """Hard sigmoid activation function: rho(s) = max(0, min(1, s))"""
    return torch.clamp(s, 0., 1.)

class Network:
    """PyTorch implementation of Equilibrium Propagation Network"""
    
    def __init__(self, name, hyperparameters=None, device='cpu'):
        """
        Initialize the network
        
        Args:
            name (str): Name of the network (used for saving/loading)
            hyperparameters (dict): Network hyperparameters
            device (str): Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        self.path = name + ".save"
        
        # Default hyperparameters
        default_hyperparams = {
            "hidden_sizes": [500],
            "n_epochs": 25,
            "batch_size": 20,
            "n_it_neg": 20,
            "n_it_pos": 4,
            "epsilon": 0.5,
            "beta": 0.5,
            "alphas": [0.1, 0.05]
        }
        
        if hyperparameters is None:
            hyperparameters = {}
        
        # Update default hyperparameters with provided ones
        for key, value in hyperparameters.items():
            default_hyperparams[key] = value
        
        self.hyperparameters = default_hyperparams
        
        # Load or initialize parameters
        self.biases, self.weights, self.training_curves = self._load_params()
        
        # Load external world (data)
        self.external_world = External_World()
        
        # Initialize persistent particles
        dataset_size = self.external_world.size_dataset
        layer_sizes = [28*28] + self.hyperparameters["hidden_sizes"] + [10]
        
        self.persistent_particles = []
        for layer_size in layer_sizes[1:]:
            particles = torch.zeros(dataset_size, layer_size, device=device)
            self.persistent_particles.append(particles)
        
        # Current mini-batch index
        self.current_index = 0
        
        print(f"Network initialized: {name}")
        print(f"Architecture: 784-{'-'.join(map(str, self.hyperparameters['hidden_sizes']))}-10")
        print(f"Device: {device}")
    
    def _load_params(self):
        """Load or initialize network parameters"""
        layer_sizes = [28*28] + self.hyperparameters["hidden_sizes"] + [10]
        
        if os.path.isfile(self.path):
            print(f"Loading parameters from {self.path}")
            try:
                with open(self.path, 'rb') as f:
                    biases_values, weights_values, training_curves = pickle.load(f)
            except:
                print("Failed to load parameters, initializing new ones")
                biases_values, weights_values, training_curves = self._initialize_params(layer_sizes)
        else:
            print("Initializing new parameters")
            biases_values, weights_values, training_curves = self._initialize_params(layer_sizes)
        
        # Convert to PyTorch tensors
        biases = []
        for bias_val in biases_values:
            bias = torch.tensor(bias_val, device=self.device, requires_grad=True)
            biases.append(bias)
        
        weights = []
        for weight_val in weights_values:
            weight = torch.tensor(weight_val, device=self.device, requires_grad=True)
            weights.append(weight)
        
        return biases, weights, training_curves
    
    def _initialize_params(self, layer_sizes):
        """Initialize network parameters using Glorot initialization"""
        # Initialize biases to zero
        biases_values = [np.zeros(size, dtype=np.float32) for size in layer_sizes]
        
        # Initialize weights using Glorot/Xavier initialization
        weights_values = []
        for size_pre, size_post in zip(layer_sizes[:-1], layer_sizes[1:]):
            bound = np.sqrt(6. / (size_pre + size_post))
            W_values = np.random.uniform(
                low=-bound,
                high=bound,
                size=(size_pre, size_post)
            ).astype(np.float32)
            weights_values.append(W_values)
        
        training_curves = {
            "training_error": [],
            "validation_error": []
        }
        
        return biases_values, weights_values, training_curves
    
    def save_params(self):
        """Save network parameters"""
        biases_values = [b.detach().cpu().numpy() for b in self.biases]
        weights_values = [W.detach().cpu().numpy() for W in self.weights]
        
        with open(self.path, 'wb') as f:
            pickle.dump((biases_values, weights_values, self.training_curves), f)
        
        print(f"Parameters saved to {self.path}")
    
    def change_mini_batch_index(self, new_index):
        """Change the current mini-batch index"""
        self.current_index = new_index
    
    def get_current_layers(self):
        """Get current layers based on mini-batch index"""
        batch_size = self.hyperparameters["batch_size"]
        start_idx = self.current_index * batch_size
        end_idx = (self.current_index + 1) * batch_size
        
        # Get input data
        x_data = self.external_world.x[start_idx:end_idx].to(self.device)
        
        # Get persistent particles for current mini-batch
        layers = [x_data]
        for particles in self.persistent_particles:
            layer = particles[start_idx:end_idx].clone().detach().requires_grad_(True)
            layers.append(layer)
        
        return layers
    
    def get_current_targets(self):
        """Get current targets based on mini-batch index"""
        batch_size = self.hyperparameters["batch_size"]
        start_idx = self.current_index * batch_size
        end_idx = (self.current_index + 1) * batch_size
        
        y_data = self.external_world.y[start_idx:end_idx].to(self.device)
        y_data_one_hot = F.one_hot(y_data, num_classes=10).float()
        
        return y_data, y_data_one_hot
    
    def energy(self, layers):
        """Compute energy function E for the current state"""
        # Squared norm term: sum(rho(layer)^2) / 2
        squared_norm = sum([torch.sum(rho(layer) ** 2, dim=1) for layer in layers]) / 2.
        
        # Linear terms: -sum(rho(layer) * bias)
        linear_terms = -sum([torch.sum(rho(layer) * bias, dim=1) 
                           for layer, bias in zip(layers, self.biases)])
        
        # Quadratic terms: -sum(rho(pre) @ W @ rho(post))
        quadratic_terms = -sum([torch.sum(rho(pre) @ W * rho(post), dim=1)
                               for pre, W, post in zip(layers[:-1], self.weights, layers[1:])])
        
        return squared_norm + linear_terms + quadratic_terms
    
    def cost(self, layers):
        """Compute cost function C (mean squared error)"""
        _, y_data_one_hot = self.get_current_targets()
        return torch.sum((layers[-1] - y_data_one_hot) ** 2, dim=1)
    
    def total_energy(self, layers, beta):
        """Compute total energy F = E + beta * C"""
        return self.energy(layers) + beta * self.cost(layers)
    
    def measure(self):
        """Measure energy, cost, and error for current state"""
        layers = self.get_current_layers()
        y_data, y_data_one_hot = self.get_current_targets()
        
        E = torch.mean(self.energy(layers))
        C = torch.mean(self.cost(layers))
        
        # Compute error rate
        y_prediction = torch.argmax(layers[-1], dim=1)
        error = torch.mean((y_prediction != y_data).float())
        
        return E.item(), C.item(), error.item()
    
    def free_phase(self, n_iterations, epsilon):
        """Run free phase dynamics"""
        layers = self.get_current_layers()
        
        for _ in range(n_iterations):
            # Compute energy and gradients
            E_sum = torch.sum(self.energy(layers))
            
            # Compute gradients with respect to layers (except input layer)
            grads = torch.autograd.grad(E_sum, layers[1:], create_graph=True)
            
            # Update layers (except input which is clamped)
            new_layers = [layers[0]]  # Keep input layer
            for i, grad in enumerate(grads):
                new_layer = rho(layers[i+1] - epsilon * grad)
                new_layer.requires_grad_(True)
                new_layers.append(new_layer)
            
            layers = new_layers
        
        # Update persistent particles
        batch_size = self.hyperparameters["batch_size"]
        start_idx = self.current_index * batch_size
        end_idx = (self.current_index + 1) * batch_size
        
        for i, (particles, layer) in enumerate(zip(self.persistent_particles, layers[1:])):
            particles[start_idx:end_idx] = layer.detach()
    
    def weakly_clamped_phase(self, n_iterations, epsilon, beta, *alphas):
        """Run weakly clamped phase and update parameters"""
        layers = self.get_current_layers()
        
        # Store initial state for computing gradients
        layers_free = [layer.clone() for layer in layers]
        
        # Run dynamics
        for _ in range(n_iterations):
            # Compute total energy and gradients
            F_sum = torch.sum(self.total_energy(layers, beta))
            
            # Compute gradients with respect to layers (except input layer)
            grads = torch.autograd.grad(F_sum, layers[1:], create_graph=True)
            
            # Update layers (except input which is clamped)
            new_layers = [layers[0]]  # Keep input layer
            for i, grad in enumerate(grads):
                new_layer = rho(layers[i+1] - epsilon * grad)
                new_layer.requires_grad_(True)
                new_layers.append(new_layer)
            
            layers = new_layers
        
        layers_weakly_clamped = layers
        
        # Compute parameter updates
        E_mean_free = torch.mean(self.energy(layers_free))
        E_mean_weakly_clamped = torch.mean(self.energy(layers_weakly_clamped))
        
        energy_diff = (E_mean_weakly_clamped - E_mean_free) / beta
        
        # Compute gradients with respect to parameters
        bias_grads = torch.autograd.grad(energy_diff, self.biases, retain_graph=True)
        weight_grads = torch.autograd.grad(energy_diff, self.weights)
        
        # Update parameters
        Delta_log = []
        
        with torch.no_grad():
            # Update biases (skip input layer bias)
            for i, (bias, grad, alpha) in enumerate(zip(self.biases[1:], bias_grads[1:], alphas)):
                bias_new = bias - alpha * grad
                Delta_log_b = torch.sqrt(torch.mean((bias_new - bias) ** 2)) / torch.sqrt(torch.mean(bias ** 2))
                bias.copy_(bias_new)
            
            # Update weights
            for i, (weight, grad, alpha) in enumerate(zip(self.weights, weight_grads, alphas)):
                weight_new = weight - alpha * grad
                Delta_log_w = torch.sqrt(torch.mean((weight_new - weight) ** 2)) / torch.sqrt(torch.mean(weight ** 2))
                Delta_log.append(Delta_log_w.item())
                weight.copy_(weight_new)
        
        return Delta_log
