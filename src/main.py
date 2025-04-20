import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from initialization import initialize
from calculation import NetworkCalculation
from visualize import plot_xor_decision_boundary

# Hyperparameters
LEARNING_RATE = 0.1
INPUT_NODES = 2
HIDDEN_NODES = 2
OUTPUT_NODES = 1
EPOCHS = 'target'  # 'target' means train until convergence
CONVERGENCE_THRESHOLD = 0.01  # Stop training when loss is below this value
MODELS_DIR = "../models"
VISUALIZATION_INTERVAL = 100
VISUALIZATION_SPEED = 0.01  # Pause duration in seconds (lower = faster animation)

# Anti-stuck parameters
PLATEAU_PATIENCE = 500  # How many epochs with no improvement before taking action
LEARNING_RATE_DECAY = 0.95  # Factor to multiply learning rate by when stuck
MIN_LEARNING_RATE = 0.001  # Minimum learning rate before reinitializing
MAX_ATTEMPTS = 3  # Maximum number of weight reinitialization attempts

# Function to setup the real-time visualization
def setup_visualization():
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(8, 6))
    
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('XOR Decision Boundary')
    
    plt.tight_layout()
    return fig, ax

# Function to update the visualization
def update_visualization(network, epoch):
    # Access the global figure and axes
    global fig, ax
    
    # Clear previous plot
    ax.clear()
    
    # Update decision boundary
    plot_xor_decision_boundary(network, ax)
    ax.set_title(f'XOR Decision Boundary (Epoch {epoch})')
    
    # Draw and pause to update the figure
    fig.canvas.draw()
    plt.pause(VISUALIZATION_SPEED)  # Use the speed parameter here

# Function to reset the model
def reset_model():
    for file_path in (os.path.join(MODELS_DIR, file) for file in os.listdir(MODELS_DIR)):
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("Model files deleted.")
    return True

# Main function to handle model training, loading and testing
def main():
    global fig, ax
    
    # Check if a saved model exists
    if os.path.exists(os.path.join(MODELS_DIR, 'weights_input_hidden.npy')):
        print("Found saved model. Loading...")
        network = NetworkCalculation.load_model(MODELS_DIR)
        
        # Test the loaded model
        print("\nTesting the loaded model:")
        test_results = network.test_model()
        for input_tuple, result in test_results.items():
            print(f"Input: {input_tuple}, Target: {result['target']}, " 
                  f"Predicted: {result['prediction']:.6f}, "
                  f"Binary: {result['binary_prediction']}, "
                  f"Correct: {result['correct']}")
        
    else:
        print("No saved model found. Training a new model...")
        
        # Initialize weights and biases
        weights_input_hidden, weights_hidden_output = initialize.weights(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)
        bias_hidden, bias_output = initialize.bias(HIDDEN_NODES, OUTPUT_NODES)

        print(f"Initializing input-hidden weights shape: {weights_input_hidden.shape}")
        print(f"Initializing hidden-output weights shape: {weights_hidden_output.shape}")
        print(f"Initializing hidden bias shape: {bias_hidden.shape}")
        print(f"Initializing output bias shape: {bias_output.shape}")
        print(f"Dataset: {initialize.dataset}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Input nodes: {INPUT_NODES}")
        print(f"Hidden nodes: {HIDDEN_NODES}")
        print(f"Output nodes: {OUTPUT_NODES}")
        print(f"Training mode: {'until convergence' if EPOCHS == 'target' else f'for {EPOCHS} epochs'}")
        if EPOCHS == 'target':
            print(f"Convergence threshold: {CONVERGENCE_THRESHOLD}")
        print()

        # Create an instance of NetworkCalculation
        network = NetworkCalculation(weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

        # Setup visualization
        fig, ax = setup_visualization()
        
        # Initial visualization before training
        update_visualization(network, 0)

        # Train the model using the new train method
        network.train(
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            visualization_func=update_visualization,
            visualization_interval=VISUALIZATION_INTERVAL,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            plateau_patience=PLATEAU_PATIENCE,
            learning_rate_decay=LEARNING_RATE_DECAY,
            min_learning_rate=MIN_LEARNING_RATE,
            max_attempts=MAX_ATTEMPTS
        )

        # Final visualization
        update_visualization(network, "Final")
        plt.ioff()  # Turn off interactive mode
        
        # Wait for user to close the plot
        print("Close the visualization window to continue...")
        plt.show(block=True)

        # Test the trained model
        print("\nTesting the trained model:")
        test_results = network.test_model()
        for input_tuple, result in test_results.items():
            print(f"Input: {input_tuple}, Target: {result['target']}, " 
                  f"Predicted: {result['prediction']:.6f}, "
                  f"Binary: {result['binary_prediction']}, "
                  f"Correct: {result['correct']}")
        
        # Save the trained model
        network.save_model(MODELS_DIR)

    # Interactive prediction mode
    print("\nInteractive prediction mode (enter 'q' to quit and 'reset' to reset the current model):")
    while True:
        try:
            user_input = input("\nEnter two binary values (0 or 1) separated by a space (e.g., '1 0'): ")
            
            if user_input.lower() == 'q':
                break
            if user_input.lower() == 'reset':
                confirm_input = input("Are you sure you want to reset the model? (y/n) ")
                if confirm_input.lower() == 'y':
                    if reset_model():
                        print("Model reset. Restart the program to train a new model.")
                    break
                else:
                    print("Reset cancelled.")
                    continue
            
            # Parse input
            x1, x2 = map(int, user_input.split())
            if x1 not in [0, 1] or x2 not in [0, 1]:
                print("Please enter only binary values (0 or 1)")
                continue
                
            # Make prediction
            input_data = np.array([[x1, x2]])
            hidden_a, _ = network.hl_activation(input_data)
            output_a, _ = network.op_activation(hidden_a)
            
            prediction = output_a[0][0]
            binary_prediction = 1 if prediction > 0.5 else 0
            
            print(f"Input: ({x1}, {x2}), Predicted: {prediction:.6f}, Binary output: {binary_prediction}")
            
        except ValueError:
            print("Invalid input format. Please enter two binary numbers separated by a space.")
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break

if __name__ == "__main__":
    main()
