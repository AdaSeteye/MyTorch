import numpy as np
from resampling import *

import numpy as np

class Conv2d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, in_height, in_width = A.shape
        out_height = in_height - self.kernel_size + 1
        out_width = in_width - self.kernel_size + 1
        
        Z = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                A_slice = A[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                for k in range(self.out_channels):
                    Z[:, k, i, j] = np.sum(A_slice * self.W[k, :, :, :], axis=(1,2,3))
        
        Z += self.b[None, :, None, None]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, _, out_height, out_width = dLdZ.shape
        
        dLdA = np.zeros_like(self.A)
        
        for i in range(out_height):
            for j in range(out_width):
                for k in range(self.out_channels):
                    dLdA[:, :, i:i+self.kernel_size, j:j+self.kernel_size] += \
                        self.W[k, :, :, :] * dLdZ[:, k, i, j][:, None, None, None]
                    self.dLdW[k, :, :, :] += np.sum(
                        self.A[:, :, i:i+self.kernel_size, j:j+self.kernel_size] * 
                        dLdZ[:, k, i, j][:, None, None, None], axis=0
                    )
        
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))
        
        return dLdA



class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.pad = padding
        
        # Initialize Conv2d_stride1
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, 
                                             weight_init_fn, bias_init_fn)
        
        # Initialize Downsample2d
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Pad the input
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 
                          mode='constant')
        
        # Apply convolution with stride 1
        Z_stride1 = self.conv2d_stride1.forward(A_padded)
        
        # Downsample the result
        Z = self.downsample2d.forward(Z_stride1)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Upsample the gradient
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        
        # Backpropagate through Conv2d_stride1
        dLdA_padded = self.conv2d_stride1.backward(dLdZ_upsampled)
        
        # Remove padding from dLdA
        if self.pad > 0:
            dLdA = dLdA_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]
        else:
            dLdA = dLdA_padded
        
        return dLdA

    @property
    def dLdW(self):
        return self.conv2d_stride1.dLdW

    @property
    def dLdb(self):
        return self.conv2d_stride1.dLdb
