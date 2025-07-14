import torch
import numpy as np
import sys
import time
from tqdm import tqdm
from model_pytorch import Network

def train_net(net):
    """Train the equilibrium propagation network"""
    
    path = net.path
    hidden_sizes = net.hyperparameters["hidden_sizes"]
    n_epochs = net.hyperparameters["n_epochs"]
    batch_size = net.hyperparameters["batch_size"]
    n_it_neg = net.hyperparameters["n_it_neg"]
    n_it_pos = net.hyperparameters["n_it_pos"]
    epsilon = net.hyperparameters["epsilon"]
    beta = net.hyperparameters["beta"]
    alphas = net.hyperparameters["alphas"]
    
    print(f"name = {path}")
    print(f"architecture = 784-{'-'.join([str(n) for n in hidden_sizes])}-10")
    print(f"number of epochs = {n_epochs}")
    print(f"batch_size = {batch_size}")
    print(f"n_it_neg = {n_it_neg}")
    print(f"n_it_pos = {n_it_pos}")
    print(f"epsilon = {epsilon:.1f}")
    print(f"beta = {beta:.1f}")
    print(f"learning rates: {' '.join([f'alpha_W{k+1}={alpha:.3f}' for k, alpha in enumerate(alphas)])}")
    print()
    
    n_batches_train = 50000 // batch_size
    n_batches_valid = 10000 // batch_size
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        
        ### TRAINING ###
        
        # Cumulative sum of training energy, training cost and training error
        measures_sum = [0., 0., 0.]
        gW = [0.] * len(alphas)
        
        # Use tqdm for progress bar
        pbar = tqdm(range(n_batches_train), desc="Training")
        
        for batch_idx in pbar:
            # Change the index of the mini batch
            net.change_mini_batch_index(batch_idx)
            
            # Free phase
            net.free_phase(n_it_neg, epsilon)
            
            # Measure the energy, cost and error at the end of the free phase
            measures = net.measure()
            measures_sum = [measure_sum + measure for measure_sum, measure in zip(measures_sum, measures)]
            measures_avg = [measure_sum / (batch_idx + 1) for measure_sum in measures_sum]
            measures_avg[-1] *= 100.  # Convert error rate to percentage
            
            # Update progress bar
            pbar.set_postfix({
                'E': f'{measures_avg[0]:.1f}',
                'C': f'{measures_avg[1]:.5f}',
                'err': f'{measures_avg[2]:.2f}%'
            })
            
            # Weakly clamped phase
            sign = 2 * np.random.randint(0, 2) - 1  # Random sign +1 or -1
            beta_signed = sign * beta
            
            Delta_logW = net.weakly_clamped_phase(n_it_pos, epsilon, beta_signed, *alphas)
            gW = [gW1 + Delta_logW1 for gW1, Delta_logW1 in zip(gW, Delta_logW)]
        
        # Print weight change statistics
        dlogW = [100. * gW1 / n_batches_train for gW1 in gW]
        print(f"   Weight changes: {' '.join([f'dlogW{k+1}={dlogW1:.3f}%' for k, dlogW1 in enumerate(dlogW)])}")
        
        net.training_curves["training_error"].append(measures_avg[-1])
        
        ### VALIDATION ###
        
        # Cumulative sum of validation energy, validation cost and validation error
        measures_sum = [0., 0., 0.]
        
        # Use tqdm for validation progress bar
        pbar = tqdm(range(n_batches_valid), desc="Validation")
        
        for batch_idx in pbar:
            # Change the index of the mini batch
            net.change_mini_batch_index(n_batches_train + batch_idx)
            
            # Free phase
            net.free_phase(n_it_neg, epsilon)
            
            # Measure the energy, cost and error at the end of the free phase
            measures = net.measure()
            measures_sum = [measure_sum + measure for measure_sum, measure in zip(measures_sum, measures)]
            measures_avg = [measure_sum / (batch_idx + 1) for measure_sum in measures_sum]
            measures_avg[-1] *= 100.  # Convert error rate to percentage
            
            # Update progress bar
            pbar.set_postfix({
                'E': f'{measures_avg[0]:.1f}',
                'C': f'{measures_avg[1]:.5f}',
                'err': f'{measures_avg[2]:.2f}%'
            })
        
        net.training_curves["validation_error"].append(measures_avg[-1])
        
        duration = (time.time() - start_time) / 60.
        print(f"   Training error: {net.training_curves['training_error'][-1]:.2f}%")
        print(f"   Validation error: {net.training_curves['validation_error'][-1]:.2f}%")
        print(f"   Duration: {duration:.1f} min")
        print()
        
        # Save the parameters of the network at the end of the epoch
        net.save_params()

def main():
    """Main function to run training"""
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Hyperparameters for different network configurations
    net_configs = {
        "net1": {
            "hidden_sizes": [500],
            "n_epochs": 25,
            "batch_size": 20,
            "n_it_neg": 20,
            "n_it_pos": 4,
            "epsilon": 0.5,
            "beta": 0.5,
            "alphas": [0.1, 0.05]
        },
        "net2": {
            "hidden_sizes": [500, 500],
            "n_epochs": 60,
            "batch_size": 20,
            "n_it_neg": 150,
            "n_it_pos": 6,
            "epsilon": 0.5,
            "beta": 1.0,
            "alphas": [0.4, 0.1, 0.01]
        },
        "net3": {
            "hidden_sizes": [500, 500, 500],
            "n_epochs": 500,
            "batch_size": 20,
            "n_it_neg": 500,
            "n_it_pos": 8,
            "epsilon": 0.5,
            "beta": 1.0,
            "alphas": [0.128, 0.032, 0.008, 0.002]
        }
    }
    
    # Default to net1 if no argument provided
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        net_name = "net1"
    
    if net_name not in net_configs:
        print(f"Unknown network configuration: {net_name}")
        print(f"Available configurations: {list(net_configs.keys())}")
        return
    
    print(f"Training {net_name}")
    
    # Create and train network
    net = Network(net_name, net_configs[net_name], device=device)
    train_net(net)

if __name__ == "__main__":
    main()
