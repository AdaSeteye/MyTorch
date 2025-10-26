
import numpy as np
from resampling import *



class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1

        # Initialize output tensor
        Z = np.zeros((batch_size, self.out_channels, output_size))

        for c_out in range(self.out_channels):
            for c_in in range(self.in_channels):
                for i in range(output_size):
                    Z[:, c_out, i] += np.sum(
                        A[:, c_in, i:i + self.kernel_size] * self.W[c_out, c_in, :], axis=1
                    )
            Z[:, c_out, :] += self.b[c_out]  # Add bias

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        _, in_channels, kernel_size = self.W.shape
        input_size = output_size + kernel_size - 1  # Reverse calculation from forward

        # Compute gradient of loss w.r.t. bias
        self.dLdb = np.sum(dLdZ, axis=(0, 2))  # Summing across batch and width

        # Compute gradient of loss w.r.t. weights (dLdW)
        self.dLdW = np.zeros(self.W.shape)

        for c_out in range(out_channels):  # Iterate over output channels
            for c_in in range(in_channels):  # Iterate over input channels
                for k in range(kernel_size):  # Iterate over kernel size
                    # Element-wise multiplication and sum
                    self.dLdW[c_out, c_in, k] = np.sum(
                        dLdZ[:, c_out, :] * self.A[:, c_in, k:k + output_size]
                    )

        # Compute gradient of loss w.r.t. input (dLdA)
        dLdA = np.zeros((batch_size, in_channels, input_size))

        # Flip weights for convolution
        W_flipped = np.flip(self.W, axis=2)

        # Pad dLdZ with (kernel_size - 1) zeros on both sides
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (kernel_size - 1, kernel_size - 1)), mode='constant')

        for c_in in range(in_channels):  # Iterate over input channels
            for c_out in range(out_channels):  # Iterate over output channels
                for i in range(input_size):  # Iterate over input width
                    dLdA[:, c_in, i] += np.sum(
                        dLdZ_padded[:, c_out, i:i + kernel_size] * W_flipped[c_out, c_in, :], axis=1
                    )

        return dLdA


from resampling import Downsample1d

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.pad = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize weights and biases
        self.W = weight_init_fn(out_channels, in_channels, kernel_size) if weight_init_fn else np.random.randn(out_channels, in_channels, kernel_size)
        self.b = bias_init_fn(out_channels) if bias_init_fn else np.zeros(out_channels)
        
        # Initialize Conv1d_stride1 and Downsample1d instances
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, 
                                             lambda *args: self.W, lambda *args: self.b)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # Pad the input
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant')
        
        # Apply convolution with stride 1
        Z_intermediate = self.conv1d_stride1.forward(A_padded)
        
        # Downsample the result
        Z = self.downsample1d.forward(Z_intermediate)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Upsample the gradient
        dLdZ_upsampled = self.downsample1d.backward(dLdZ)
        
        # Backpropagate through Conv1d_stride1
        dLdA_padded = self.conv1d_stride1.backward(dLdZ_upsampled)
        
        # Remove padding from dLdA
        if self.pad > 0:
            dLdA = dLdA_padded[:, :, self.pad:-self.pad]
        else:
            dLdA = dLdA_padded
        
        # Update weights and biases
        self.W = self.conv1d_stride1.W
        self.b = self.conv1d_stride1.b
        
        return dLdA
