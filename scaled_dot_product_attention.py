import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        self.eps = 1e10  # Large negative value for masking
        # Softmax along the last dimension (source sequence length dimension)
        self.softmax = Softmax(dim=-1)
        
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E)
        :param K: Key matrix of shape (N, ..., H, S, E)
        :param V: Value matrix of shape (N, ..., H, S, Ev)
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        d_k = Q.shape[-1]
        # Calculate scaled dot product attention scores
        scaled_dot_product = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scaled_dot_product = np.where(mask, scaled_dot_product - self.eps, scaled_dot_product)
        
        # Apply softmax to get attention weights
        self.attention_scores = self.softmax.forward(scaled_dot_product)
        
        # Calculate output as weighted sum of values
        output = np.matmul(self.attention_scores, V)
        
        # Store inputs for backward pass
        self.Q, self.K, self.V = Q, K, V
        
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradients wrt Q, K, V
        """
        # Gradient wrt V
        d_V = np.matmul(np.swapaxes(self.attention_scores, -2, -1), d_output)
        
        # Gradient wrt attention scores
        d_attention_scores = np.matmul(d_output, np.swapaxes(self.V, -2, -1))
        
        # Gradient wrt scaled dot product before softmax using Softmax.backward
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        # Scale gradients by sqrt(d_k)
        d_k = self.Q.shape[-1]
        d_scaled_dot_product /= np.sqrt(d_k)
        
        # Gradients wrt Q and K
        d_Q = np.matmul(d_scaled_dot_product, self.K)
        d_K = np.matmul(np.swapaxes(d_scaled_dot_product, -2, -1), self.Q)
        
        return d_Q, d_K, d_V
