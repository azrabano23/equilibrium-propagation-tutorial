import tkinter as tk
from tkinter import ttk
import numpy as np
import sys
import time
from threading import Thread
import torch
from PIL import Image, ImageTk
from model_pytorch import Network

class GUI:
    """Modern GUI for visualizing Equilibrium Propagation network"""
    
    def __init__(self, name):
        self.root = tk.Tk()
        self.root.title('Equilibrium Propagation - PyTorch')
        self.root.geometry('800x600')
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize network
        self.net = Network(name=name, hyperparameters={"batch_size": 1}, device=device)
        self.hidden_sizes = self.net.hyperparameters["hidden_sizes"]
        self.n_layers = len(self.hidden_sizes) + 2
        
        self.setup_ui()
        
        # Start the visualization thread
        self.running = True
        self.thread = Thread(target=self.run_visualization)
        self.thread.daemon = True
        self.thread.start()
    
    def setup_ui(self):
        """Set up the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Image index control
        ttk.Label(controls_frame, text="Image Index:").grid(row=0, column=0, padx=(0, 5))
        self.index_var = tk.StringVar(value="0")
        index_entry = ttk.Entry(controls_frame, textvariable=self.index_var, width=10)
        index_entry.grid(row=0, column=1, padx=(0, 20))
        
        # Control buttons
        ttk.Button(controls_frame, text="Random", command=self.random_image).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(controls_frame, text="Next", command=self.next_image).grid(row=0, column=3, padx=(0, 5))
        ttk.Button(controls_frame, text="Previous", command=self.prev_image).grid(row=0, column=4, padx=(0, 5))
        
        # Canvas frame
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for visualization
        self.canvas = tk.Canvas(canvas_frame, width=700, height=500, bg='white')
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for canvas
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status labels
        self.energy_label = ttk.Label(status_frame, text="Energy: --")
        self.energy_label.grid(row=0, column=0, padx=(0, 20))
        
        self.cost_label = ttk.Label(status_frame, text="Cost: --")
        self.cost_label.grid(row=0, column=1, padx=(0, 20))
        
        self.prediction_label = ttk.Label(status_frame, text="Prediction: --")
        self.prediction_label.grid(row=0, column=2, padx=(0, 20))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Initial canvas setup
        self.update_canvas(first_time=True)
    
    def get_current_index(self):
        """Get the current image index"""
        try:
            index = int(self.index_var.get())
            # Map to test set indices (60000-69999)
            return (index % 10000) + 60000
        except ValueError:
            return 60000
    
    def random_image(self):
        """Select a random image"""
        random_idx = np.random.randint(0, 10000)
        self.index_var.set(str(random_idx))
    
    def next_image(self):
        """Go to next image"""
        try:
            current = int(self.index_var.get())
            self.index_var.set(str((current + 1) % 10000))
        except ValueError:
            self.index_var.set("0")
    
    def prev_image(self):
        """Go to previous image"""
        try:
            current = int(self.index_var.get())
            self.index_var.set(str((current - 1) % 10000))
        except ValueError:
            self.index_var.set("0")
    
    def update_canvas(self, first_time=False):
        """Update the canvas with current network state"""
        try:
            # Get current layers
            layers = self.net.get_current_layers()
            
            # Calculate dimensions for visualization
            units = [(28, 28)] + [(10, size // 10) for size in self.hidden_sizes] + [(1, 10)]
            pixels = [(140, 140)] + [(size // 2, 50) for size in self.hidden_sizes] + [(250, 25)]
            
            # Convert layers to numpy arrays and normalize
            arrays = []
            for layer, dimensions in zip(layers, units):
                array = layer.detach().cpu().numpy().reshape(dimensions)
                # Normalize to 0-255 range
                array = (array - array.min()) / (array.max() - array.min() + 1e-8)
                array = (array * 255).astype(np.uint8)
                arrays.append(array)
            
            # Create images
            images = []
            for array, pixel_dims in zip(arrays, pixels):
                img = Image.fromarray(array, mode='L')
                img = img.resize(pixel_dims, Image.NEAREST)
                images.append(img)
            
            # Convert to PhotoImage
            self.photo_images = [ImageTk.PhotoImage(img) for img in images]
            
            # Get measurements
            energy, cost, error = self.net.measure()
            
            # Get prediction
            prediction = torch.argmax(layers[-1], dim=1).item()
            
            if first_time:
                # Create canvas items
                self.canvas.delete("all")
                
                # Draw images
                self.img_canvas = []
                y_pos = 50
                for i, photo_img in enumerate(self.photo_images):
                    img_id = self.canvas.create_image(350, y_pos, image=photo_img)
                    self.img_canvas.append(img_id)
                    
                    # Add layer label
                    if i == 0:
                        label = "Input Layer"
                    elif i == len(self.photo_images) - 1:
                        label = "Output Layer"
                    else:
                        label = f"Hidden Layer {i}"
                    
                    self.canvas.create_text(50, y_pos, text=label, anchor="w", font=("Arial", 12, "bold"))
                    y_pos += 100
                
                # Update scroll region
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            else:
                # Update existing images
                for img_canvas, photo_img in zip(self.img_canvas, self.photo_images):
                    self.canvas.itemconfig(img_canvas, image=photo_img)
            
            # Update status labels
            self.energy_label.config(text=f"Energy: {energy:.2f}")
            self.cost_label.config(text=f"Cost: {cost:.4f}")
            self.prediction_label.config(text=f"Prediction: {prediction}")
            
        except Exception as e:
            print(f"Error updating canvas: {e}")
    
    def run_visualization(self):
        """Run the visualization loop"""
        while self.running:
            try:
                # Update network index
                current_index = self.get_current_index()
                self.net.change_mini_batch_index(current_index)
                
                # Run one iteration of free phase
                self.net.free_phase(n_iterations=1, epsilon=0.1)
                
                # Update canvas
                self.update_canvas()
                
                # Sleep for smooth animation
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in visualization loop: {e}")
                time.sleep(1)
    
    def run(self):
        """Start the GUI main loop"""
        try:
            self.root.mainloop()
        finally:
            self.running = False

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python gui_pytorch.py <network_name>")
        print("Example: python gui_pytorch.py net1")
        sys.exit(1)
    
    name = sys.argv[1]
    
    # Check if the network file exists
    if not os.path.exists(name + ".save"):
        print(f"Network file {name}.save not found!")
        print("Please train a network first using train_model_pytorch.py")
        sys.exit(1)
    
    try:
        gui = GUI(name)
        gui.run()
    except KeyboardInterrupt:
        print("\nGUI interrupted by user")
    except Exception as e:
        print(f"Error running GUI: {e}")

if __name__ == "__main__":
    import os
    main()
