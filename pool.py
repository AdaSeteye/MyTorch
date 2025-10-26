import numpy as np 
from resampling import *

import numpy as np

class MaxPool2d_stride1:
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        N, C, H, W = A.shape
        H_out = H - self.kernel + 1
        W_out = W - self.kernel + 1
        
        Z = np.zeros((N, C, H_out, W_out))
        self.max_indices = np.zeros((N, C, H_out, W_out, 2), dtype=int)
        
        for i in range(H_out):
            for j in range(W_out):
                window = A[:, :, i:i+self.kernel, j:j+self.kernel]
                Z[:, :, i, j] = np.max(window, axis=(2, 3))
                max_pos = np.unravel_index(window.reshape(N, C, -1).argmax(axis=2), (self.kernel, self.kernel))
                self.max_indices[:, :, i, j, 0] = max_pos[0] + i
                self.max_indices[:, :, i, j, 1] = max_pos[1] + j
        
        return Z

    def backward(self, dLdZ):
        N, C, H_out, W_out = dLdZ.shape
        dLdA = np.zeros((N, C, H_out + self.kernel - 1, W_out + self.kernel - 1))
        
        for i in range(H_out):
            for j in range(W_out):
                max_i, max_j = self.max_indices[:, :, i, j, 0], self.max_indices[:, :, i, j, 1]
                dLdA[np.arange(N)[:, None], np.arange(C)[None, :], max_i, max_j] += dLdZ[:, :, i, j]
        
        return dLdA


class MeanPool2d_stride1:
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        N, C, H, W = A.shape
        H_out = H - self.kernel + 1
        W_out = W - self.kernel + 1
        
        Z = np.zeros((N, C, H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                Z[:, :, i, j] = np.mean(A[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))
        
        return Z

    def backward(self, dLdZ):
        N, C, H_out, W_out = dLdZ.shape
        H = H_out + self.kernel - 1
        W = W_out + self.kernel - 1
        dLdA = np.zeros((N, C, H, W))
        
        for i in range(H_out):
            for j in range(W_out):
                dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += dLdZ[:, :, i, j][:, :, None, None] / (self.kernel * self.kernel)
        
        return dLdA


class MaxPool2d:
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        Z = self.maxpool2d_stride1.forward(A)
        return self.downsample2d.forward(Z)

    def backward(self, dLdZ):
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        return self.maxpool2d_stride1.backward(dLdZ_upsampled)

class MeanPool2d:
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        Z = self.meanpool2d_stride1.forward(A)
        return self.downsample2d.forward(Z)

    def backward(self, dLdZ):
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        return self.meanpool2d_stride1.backward(dLdZ_upsampled)
