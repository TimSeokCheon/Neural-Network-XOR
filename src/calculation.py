import numpy as np
from initialization import initialize
import os
import matplotlib.pyplot as plt

class NetworkCalculation:
    def __init__(self, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
        self.weights_input_hidden = weights_input_hidden
        self.weights_hidden_output = weights_hidden_output
        self.bias_hidden = bias_hidden
        self.bias_output = bias_output
        
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        # Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        # Since we're passing in the sigmoid output directly, we use:
        return x * (1 - x)
        
    @staticmethod
    def fetch_and_convert_input(input_example=(1, 0)): # Get a single input from the dataset
        target = initialize.dataset[input_example]

        # Convert input to numpy array for matrix operations
        x = np.array([input_example])  # Shape: (1, 2)

        return x, target

    # Forward pass through the network

    def hl_activation(self, input_data):# Calculate hidden layer activations
        hidden_z = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden  # Shape: (1, hidden_nodes)
        hidden_a = self.sigmoid(hidden_z)  # Apply activation function

        return hidden_a, hidden_z

    def op_activation(self, hidden_a):  # Updated to accept hidden_a as parameter
        # Calculate output layer activations
        output_z = np.dot(hidden_a, self.weights_hidden_output) + self.bias_output  # Shape: (1, 1)
        output_a = self.sigmoid(output_z)  # Final prediction

        return output_a, output_z
    

    # Calculate error/loss
    def BCE_loss(self, output_a, target):  # Updated to accept parameters
        loss = -((target * np.log(output_a[0][0])) + ((1 - target) * np.log(1 - output_a[0][0])))
        
        return loss

    def backpropagation(self, input_data, hidden_a, output_a, target, learning_rate):
        """
        Perform backpropagation to update weights and biases.
        
        Parameters:
        - input_data: Input values
        - hidden_a: Hidden layer activations
        - output_a: Output layer activation (prediction)
        - target: Target value
        - learning_rate: Learning rate for gradient descent
        """
        # Convert target to array if it's a scalar
        if np.isscalar(target):
            target = np.array([[target]])
            
        # Calculate gradients for output layer
        output_error = output_a - target  # Gradient of loss with respect to output
        
        # Gradients for weights between hidden and output layers
        grad_weights_hidden_output = hidden_a.T @ output_error
        
        # Gradient for output bias
        grad_bias_output = np.sum(output_error, axis=0, keepdims=True)
        
        # Calculate gradients for hidden layer
        # Error propagated back to hidden layer
        hidden_error = output_error @ self.weights_hidden_output.T
        
        # Apply derivative of activation function
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_a)
        
        # Gradients for weights between input and hidden layers
        grad_weights_input_hidden = input_data.T @ hidden_delta
        
        # Gradient for hidden bias
        grad_bias_hidden = np.sum(hidden_delta, axis=0, keepdims=True)
        
        # Update weights and biases using gradient descent
        self.weights_hidden_output -= learning_rate * grad_weights_hidden_output
        self.bias_output -= learning_rate * grad_bias_output
        self.weights_input_hidden -= learning_rate * grad_weights_input_hidden
        self.bias_hidden -= learning_rate * grad_bias_hidden
        
        return {
            'grad_weights_hidden_output': grad_weights_hidden_output,
            'grad_bias_output': grad_bias_output,
            'grad_weights_input_hidden': grad_weights_input_hidden,
            'grad_bias_hidden': grad_bias_hidden
        }
    
    def save_model(self, directory="../models"):
        """
        Save the trained model parameters to disk.
        
        Parameters:
        - directory: Directory to save the model parameters
        """
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save each parameter to a separate file
        np.save(os.path.join(directory, 'weights_input_hidden.npy'), self.weights_input_hidden)
        np.save(os.path.join(directory, 'weights_hidden_output.npy'), self.weights_hidden_output)
        np.save(os.path.join(directory, 'bias_hidden.npy'), self.bias_hidden)
        np.save(os.path.join(directory, 'bias_output.npy'), self.bias_output)
        
        print(f"Model saved to {directory}")
    
    @classmethod
    def load_model(cls, directory="../models"):
        """
        Load the trained model parameters from disk.
        
        Parameters:
        - directory: Directory where the model parameters are stored
        
        Returns:
        - An instance of NetworkCalculation with loaded parameters
        """
        try:
            weights_input_hidden = np.load(os.path.join(directory, 'weights_input_hidden.npy'))
            weights_hidden_output = np.load(os.path.join(directory, 'weights_hidden_output.npy'))
            bias_hidden = np.load(os.path.join(directory, 'bias_hidden.npy'))
            bias_output = np.load(os.path.join(directory, 'bias_output.npy'))
            
            print(f"Model loaded from {directory}")
            return cls(weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
        except FileNotFoundError:
            print("Model files not found. Please train and save the model first.")
            return None
    
    def train_one_epoch(self, learning_rate=0.1):
        """
        Train the model for one epoch over all examples in the dataset.
        
        Parameters:
        - learning_rate: Learning rate for gradient descent
        
        Returns:
        - epoch_loss: Total loss for this epoch
        """
        epoch_loss = 0
        
        # Train on each example in the dataset
        for input_tuple, target in initialize.dataset.items():
            # Convert input to numpy array
            input_data = np.array([input_tuple])
            
            # Forward pass
            hidden_a, hidden_z = self.hl_activation(input_data)
            output_a, output_z = self.op_activation(hidden_a)
            
            # Calculate loss
            loss = self.BCE_loss(output_a, target)
            epoch_loss += loss
            
            # Backward pass - update weights and biases
            self.backpropagation(input_data, hidden_a, output_a, target, learning_rate)
            
        return epoch_loss
    
    def reinitialize_weights(self, input_nodes, hidden_nodes, output_nodes):
        """
        Reinitialize the network weights when training gets stuck.
        Returns True if weights were reinitialized.
        """
        print("\nTraining appears to be stuck. Reinitializing weights...")
        self.weights_input_hidden, self.weights_hidden_output = initialize.weights(input_nodes, hidden_nodes, output_nodes)
        self.bias_hidden, self.bias_output = initialize.bias(hidden_nodes, output_nodes)
        return True
    
    def train(self, epochs='target', learning_rate=0.1, visualization_func=None, 
              visualization_interval=100, convergence_threshold=0.01, plateau_patience=500,
              learning_rate_decay=0.95, min_learning_rate=0.001, max_attempts=3):
        """
        Train the model with anti-stuck mechanisms.
        
        Parameters:
        - epochs: Number of epochs to train, or 'target' to train until convergence
        - learning_rate: Learning rate for gradient descent
        - visualization_func: Function to update visualization during training
        - visualization_interval: How often to update visualization
        - convergence_threshold: Threshold for early stopping when loss is below this value
        - plateau_patience: Number of epochs to wait before considering training stuck
        - learning_rate_decay: Factor to multiply learning rate when stuck
        - min_learning_rate: Minimum learning rate before reinitializing
        - max_attempts: Maximum number of reinitialization attempts
        
        Returns:
        - training_history: List of average losses during training
        """
        training_history = []
        epoch = 0
        converged = False
        best_loss = float('inf')
        plateau_counter = 0
        current_lr = learning_rate
        attempt = 1
        
        # Get shapes for potential reinitialization
        input_nodes = self.weights_input_hidden.shape[0]
        hidden_nodes = self.weights_input_hidden.shape[1]
        output_nodes = self.weights_hidden_output.shape[1]
        
        # Determine the maximum number of epochs
        max_epochs = 20000  # Default maximum if training until convergence
        if epochs != 'target':
            max_epochs = epochs
        
        while epoch < max_epochs and not converged and attempt <= max_attempts:
            epoch += 1
            epoch_loss = self.train_one_epoch(current_lr)
            avg_loss = epoch_loss / len(initialize.dataset)
            training_history.append(avg_loss)
            
            # Update visualization if provided
            if visualization_func and epoch % visualization_interval == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}, Learning rate: {current_lr:.6f}")
                visualization_func(self, epoch)
            
            # Check for convergence if training until target
            if epochs == 'target' and avg_loss < convergence_threshold:
                print(f"\nConverged at epoch {epoch} with loss {avg_loss:.6f}")
                converged = True
                
            # Check for improvement
            if avg_loss < best_loss - 0.00001:  # Small epsilon to account for floating point errors
                best_loss = avg_loss
                plateau_counter = 0
            else:
                plateau_counter += 1
            
            # Handle plateaus
            if plateau_counter >= plateau_patience:
                if current_lr > min_learning_rate:
                    # Try reducing learning rate first
                    current_lr *= learning_rate_decay
                    print(f"\nLoss plateaued. Reducing learning rate to {current_lr:.6f}")
                    plateau_counter = 0
                else:
                    if attempt < max_attempts:
                        # Reinitialize weights
                        self.reinitialize_weights(input_nodes, hidden_nodes, output_nodes)
                        current_lr = learning_rate  # Reset learning rate
                        plateau_counter = 0
                        attempt += 1
                        print(f"Attempt {attempt}/{max_attempts}")
                    else:
                        print("\nMaximum attempts reached. Stopping training.")
                        break
        
        if not converged and epochs == 'target':
            if attempt >= max_attempts:
                print(f"\nDid not converge after {max_attempts} attempts. Best loss: {best_loss:.6f}")
            else:
                print(f"\nDid not converge after {epoch} epochs. Final loss: {avg_loss:.6f}")
        else:
            if converged:
                print(f"\nTraining successfully converged after {epoch} epochs.")
            else:
                print(f"\nTraining complete after {epoch} epochs.")
            
        return training_history
    
    def test_model(self):
        """
        Test the model on all examples in the dataset.
        
        Returns:
        - results: Dictionary with test results for each input
        """
        results = {}
        for input_tuple, target in initialize.dataset.items():
            input_data = np.array([input_tuple])
            hidden_a, _ = self.hl_activation(input_data)
            output_a, _ = self.op_activation(hidden_a)
            prediction = output_a[0][0]
            binary_prediction = 1 if prediction > 0.5 else 0
            correct = binary_prediction == target
            
            results[input_tuple] = {
                'target': target,
                'prediction': prediction,
                'binary_prediction': binary_prediction,
                'correct': correct
            }
            
        return results