import numpy as np

class initialize:
    # Data
    dataset = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0
    }
    @staticmethod
    def weights(input_nodes, hidden_nodes, output_nodes):
        """
        Initialize weights for a neural network with the given number of input, hidden, and output nodes.
        
        Parameters:
        - input_nodes: Number of input nodes
        - hidden_nodes: Number of hidden nodes
        - output_nodes: Number of output nodes
        
        Returns:
        - weights_input_hidden: Matrix of weights between input and hidden layers
        - weights_hidden_output: Matrix of weights between hidden and output layers
        """
        # Initialize weights using Xavier/Glorot initialization
        # This creates a proper weight matrix for each layer connection
        weights_input_hidden = np.random.randn(input_nodes, hidden_nodes) * np.sqrt(2 / (input_nodes + hidden_nodes))
        weights_hidden_output = np.random.randn(hidden_nodes, output_nodes) * np.sqrt(2 / (hidden_nodes + output_nodes))
        
        return weights_input_hidden, weights_hidden_output
    
    @staticmethod
    def bias(hidden_nodes, output_nodes):
        """
        Initialize biases for hidden and output layers.
        
        Parameters:
        - hidden_nodes: Number of hidden nodes
        - output_nodes: Number of output nodes
        
        Returns:
        - bias_hidden: Bias values for hidden layer
        - bias_output: Bias values for output layer
        """
        # Initialize biases with zeros
        bias_hidden = np.zeros((1, hidden_nodes))
        bias_output = np.zeros((1, output_nodes))
        
        return bias_hidden, bias_output