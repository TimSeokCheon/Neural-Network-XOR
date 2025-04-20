import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import sys
from calculation import NetworkCalculation

def plot_xor_decision_boundary(network, ax=None, resolution=100):
    """
    Visualize the decision boundary of the neural network for the XOR function.
    
    Parameters:
    - network: Trained NetworkCalculation instance
    - ax: Matplotlib axis for plotting
    - resolution: Resolution of the grid
    """
    # Create a grid of points across the input space
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    
    # Create meshgrid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Reshape to get a list of points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict using the neural network
    predictions = []
    for point in grid_points:
        input_data = np.array([point])
        hidden_a, _ = network.hl_activation(input_data)
        output_a, _ = network.op_activation(hidden_a)
        predictions.append(output_a[0][0])
    
    # Reshape back to grid format
    Z = np.array(predictions).reshape(xx.shape)
    
    # Create plot
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the decision boundary
    cmap = ListedColormap(['#FFAAAA', '#AAAAFF'])
    ax.contourf(xx, yy, Z > 0.5, alpha=0.5, cmap=cmap)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='-')
    
    # Plot the training points
    xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    xor_outputs = [0, 1, 1, 0]
    
    ax.scatter([x[0] for x in xor_inputs], 
               [x[1] for x in xor_inputs], 
               c=xor_outputs, 
               cmap=ListedColormap(['#FF0000', '#0000FF']),
               edgecolors='k', 
               s=100)
    
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return ax

def plot_hidden_activations(network, ax=None, resolution=100):
    """
    Visualize the hidden layer activations.
    
    Parameters:
    - network: Trained NetworkCalculation instance
    - ax: Matplotlib axes for plotting
    - resolution: Resolution of the grid
    """
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create a grid of points
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get hidden layer activations for each grid point
    hidden_activations = []
    for point in grid_points:
        input_data = np.array([point])
        hidden_a, _ = network.hl_activation(input_data)
        hidden_activations.append(hidden_a[0])
    
    hidden_activations = np.array(hidden_activations)
    
    # Reshape for plotting
    Z1 = hidden_activations[:, 0].reshape(xx.shape)
    Z2 = hidden_activations[:, 1].reshape(xx.shape)
    
    # Plot each hidden neuron's activation
    for i, (z, title) in enumerate(zip([Z1, Z2], ["Hidden Neuron 1", "Hidden Neuron 2"])):
        cax = ax[i].contourf(xx, yy, z, cmap='viridis', alpha=0.8)
        ax[i].contour(xx, yy, z, levels=10, colors='black', linestyles='-', linewidths=0.5)
        ax[i].set_title(title)
        ax[i].set_xlabel('Input 1')
        ax[i].set_ylabel('Input 2')
        ax[i].grid(True, linestyle='--', alpha=0.6)
        plt.colorbar(cax, ax=ax[i])
    
    return ax

def visualize_learning_process(epochs=10000, save_interval=1000, resolution=100):
    """
    Visualize how the neural network learns the XOR function over time.
    
    Parameters:
    - epochs: Total number of epochs
    - save_interval: Interval for saving visualizations
    - resolution: Resolution of the grid
    """
    from initialization import initialize
    
    # Directory to save visualizations
    vis_dir = "../visualizations"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Initialize network
    input_nodes = 2
    hidden_nodes = 2
    output_nodes = 1
    learning_rate = 0.1
    
    # Initialize weights and biases
    weights_input_hidden, weights_hidden_output = initialize.weights(input_nodes, hidden_nodes, output_nodes)
    bias_hidden, bias_output = initialize.bias(hidden_nodes, output_nodes)
    
    # Create network
    network = NetworkCalculation(weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
    
    # Training loop with visualization
    for epoch in range(epochs + 1):
        # Train on each example in the dataset
        for input_tuple, target in initialize.dataset.items():
            # Forward pass
            input_data = np.array([input_tuple])
            hidden_a, hidden_z = network.hl_activation(input_data)
            output_a, output_z = network.op_activation(hidden_a)
            
            # Backward pass
            network.backpropagation(input_data, hidden_a, output_a, target, learning_rate)
        
        # Save visualization at specified intervals
        if epoch % save_interval == 0 or epoch == epochs:
            # Create figure for decision boundary
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_xor_decision_boundary(network, ax)
            ax.set_title(f'XOR Decision Boundary (Epoch {epoch})')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'decision_boundary_epoch_{epoch}.png'))
            plt.close()
            
            # Create figure for hidden activations
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            plot_hidden_activations(network, ax)
            fig.suptitle(f'Hidden Layer Activations (Epoch {epoch})')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'hidden_activations_epoch_{epoch}.png'))
            plt.close()
            
            print(f"Saved visualization for epoch {epoch}")
    
    # Create animation from saved files
    print("Visualizations saved to", vis_dir)
    print("You can create a GIF from these images using tools like ImageMagick.")
    print("Example command: convert -delay 100 -loop 0 decision_boundary_epoch_*.png xor_learning.gif")

def visualize_trained_model():
    """
    Visualize decision boundary and hidden activations of the trained model.
    """
    # Load the trained model
    network = NetworkCalculation.load_model()
    
    if network is None:
        print("No trained model found. Please train a model first.")
        return
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Plot decision boundary
    ax1 = fig.add_subplot(1, 3, 1)
    plot_xor_decision_boundary(network, ax1)
    ax1.set_title('XOR Decision Boundary')
    
    # Plot hidden activations
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    plot_hidden_activations(network, [ax2, ax3])
    
    plt.tight_layout()
    plt.savefig("../visualizations/trained_model_visualization.png")
    plt.show()
    
    print("Visualization saved to ../visualizations/trained_model_visualization.png")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        # Visualize the learning process
        epochs = 5000  # Fewer epochs for faster visualization
        save_interval = 500
        print(f"Visualizing learning process over {epochs} epochs...")
        visualize_learning_process(epochs=epochs, save_interval=save_interval)
    else:
        # Visualize the already trained model
        visualize_trained_model()