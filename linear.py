import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A 

        self.input_shape = A.shape
        self.in_features = A.shape[-1]
        self.A_flatten = A.reshape(-1, self.in_features)
        
        Z_flatten = self.A_flatten @ self.W.T + self.b
        self.output_feature = self.W.shape[0]
        return Z_flatten.reshape(*self.input_shape[:-1], self.output_feature) 
    
    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        dLdZ_flattened = dLdZ.reshape(-1, self.output_feature) 
        # Compute gradients (refer to the equations in the writeup)
        dLdA_flatten = dLdZ_flattened @ self.W
        self.dLdW = dLdZ_flattened.T @ self.A_flatten
        self.dLdb = dLdZ_flattened.sum(axis=0)
        dLdA = dLdA_flatten.reshape(*self.input_shape)         
        # Return gradient of loss wrt input
        return dLdA
