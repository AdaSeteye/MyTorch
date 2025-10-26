import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # TODO Create a new array Z with the correct shape
        batch_size, in_channels, input_width = A.shape

        output_width = self.upsampling_factor * (input_width - 1) + 1
        Z = np.zeros((batch_size, in_channels, output_width))  # TODO
        Z[..., ::self.upsampling_factor] = A

        # TODO Fill in the values of Z by upsampling A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        # TODO Slice dLdZ by the upsampling factor to get dLdA

        dLdA = dLdZ[..., ::self.upsampling_factor]  # TODO

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_width = None  # TODO Store input width for backward computation

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # TODO Store input width
        self.input_width = A.shape[2]  # TODO

        # TODO Slice A by the downsampling factor to get Z
        Z = A[..., ::self.downsampling_factor]  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        # TODO Create a new array dLdA with the correct shape
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.input_width))  # TODO

        # TODO Fill in the values of dLdA, assigning dLdZ values at sampled positions
        dLdA[..., ::self.downsampling_factor] = dLdZ  # TODO

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        # TODO Create a new array Z with the correct shape

        batch_size, in_channels, input_height, input_width = A.shape
        output_height = self.upsampling_factor * (input_height - 1) + 1
        output_width = self.upsampling_factor * (input_width - 1) + 1

        Z = np.zeros((batch_size, in_channels, output_height, output_width))  # TODO

        # TODO Fill in the values of Z by upsampling A
        
        Z[..., ::self.upsampling_factor, ::self.upsampling_factor] = A  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # TODO Slice dLdZ by the upsampling factor to get dLdA

        dLdA = dLdZ[..., ::self.upsampling_factor, ::self.upsampling_factor]  # TODO

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_height = None  # TODO 
        self.input_width = None  # TODO 

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        # TODO Store input height and width
        self.input_height, self.input_width = A.shape[2], A.shape[3]  # TODO

        # TODO Slice A by the downsampling factor to get Z
        Z = A[..., ::self.downsampling_factor, ::self.downsampling_factor]  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # TODO Create a new array dLdA with the correct shape
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.input_height, self.input_width))  # TODO

        # TODO Fill in the values of dLdA, assigning dLdZ values at sampled positions
        dLdA[..., ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ  # TODO

        return dLdA
