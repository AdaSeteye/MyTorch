import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        
        shift_Z = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(shift_Z)
        self.A = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
       
        shape = self.A.shape
        C = shape[self.dim]
           
        if len(shape) > 2:
            A_moved = np.moveaxis(self.A, self.dim, -1)
            dLdA_moved = np.moveaxis(dLdA, self.dim, -1)
            new_shape = (-1, C)
            A_2d = A_moved.reshape(new_shape)
            dLdA_2d = dLdA_moved.reshape(new_shape)
        else:
            A_2d = self.A
            dLdA_2d = dLdA

        dLdZ_2d = np.zeros_like(dLdA_2d)

        for i in range(dLdA_2d.shape[0]):
            a = A_2d[i]
            J = np.diag(a) - np.outer(a, a)
            dLdZ_2d[i] = np.dot(dLdA_2d[i], J)

        if len(shape) > 2:
            A_moved_shape = np.moveaxis(self.A, self.dim, -1).shape
            dLdZ_moved = dLdZ_2d.reshape(A_moved_shape)
            dLdZ = np.moveaxis(dLdZ_moved, -1, self.dim)
        else:
            dLdZ = dLdZ_2d

        return dLdZ
