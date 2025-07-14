#!/usr/bin/env python3
"""
Simple text-based visualization of the Equilibrium Propagation network
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from model_pytorch import Network
import sys

def visualize_network_state(net, image_index=0):
    """Visualize the current state of the network"""
    
    # Set the network to use a specific image
    net.change_mini_batch_index(image_index + 60000)  # Test set starts at 60000
    
    # Get current layers before and after free phase
    print(f"Analyzing image {image_index}...")
    
    # Initial state
    layers_initial = net.get_current_layers()
    energy_initial, cost_initial, error_initial = net.measure()
    
    print(f"Initial state:")
    print(f"  Energy: {energy_initial:.3f}")
    print(f"  Cost: {cost_initial:.3f}")
    print(f"  Error: {error_initial:.3f}")
    
    # Run free phase
    print("Running free phase relaxation...")
    net.free_phase(n_iterations=20, epsilon=0.5)
    
    # Final state
    layers_final = net.get_current_layers()
    energy_final, cost_final, error_final = net.measure()
    
    print(f"After free phase:")
    print(f"  Energy: {energy_final:.3f}")
    print(f"  Cost: {cost_final:.3f}")
    print(f"  Error: {error_final:.3f}")
    
    # Get prediction
    prediction = torch.argmax(layers_final[-1], dim=1).item()
    actual_label = net.external_world.y[image_index + 60000].item()
    
    print(f"Prediction: {prediction}")
    print(f"Actual label: {actual_label}")
    print(f"Correct: {'Yes' if prediction == actual_label else 'No'}")
    
    # Display input image as ASCII art
    print("\nInput image (28x28 MNIST digit):")
    input_image = layers_final[0].detach().cpu().numpy().reshape(28, 28)
    
    # Convert to ASCII art
    ascii_chars = " .-+*#"
    normalized = ((input_image - input_image.min()) / (input_image.max() - input_image.min() + 1e-8))
    ascii_art = ""
    for row in normalized:
        for pixel in row:
            char_idx = int(pixel * (len(ascii_chars) - 1))
            ascii_art += ascii_chars[char_idx]
        ascii_art += "\n"
    
    print(ascii_art)
    
    # Show output layer activations
    print("Output layer activations (for digits 0-9):")
    output_activations = layers_final[-1].detach().cpu().numpy().flatten()
    for i, activation in enumerate(output_activations):
        bar_length = int(activation * 20)  # Scale to 20 characters
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {i}: {bar} {activation:.3f}")
    
    print("\n" + "="*60 + "\n")

def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        network_name = sys.argv[1]
    else:
        network_name = "net1"
    
    print(f"Loading network: {network_name}")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the network
    try:
        net = Network(name=network_name, hyperparameters={"batch_size": 1}, device=device)
        print(f"Network loaded successfully!")
        print(f"Architecture: 784-{'-'.join(map(str, net.hyperparameters['hidden_sizes']))}-10")
    except Exception as e:
        print(f"Error loading network: {e}")
        print(f"Make sure {network_name}.save exists. Train the network first if needed.")
        return
    
    # Visualize several examples
    print("\nAnalyzing several test images...\n")
    
    for i in range(5):  # Show 5 examples
        visualize_network_state(net, image_index=i * 100)  # Sample every 100th image

if __name__ == "__main__":
    main()
