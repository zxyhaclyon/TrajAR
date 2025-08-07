# import logging
import time
import os
import torch
import numpy as np
from scipy.interpolate import CubicSpline

def check_dir(path:str ,mkdir=False):
    if os.path.exists(path):
        return True
    elif mkdir:
        os.makedirs(path)
        return True
    
    return False

def get_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

class StandardScaler():
    def __init__(self, data):

        mean,std = data.float().mean(), data.float().std()

        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class MinMaxScaler():
    def __init__(self, min,max):

        self._min = min
        self._max = max

    def transform(self, data):
        data = 1. * (data - self._min)/(self._max - self._min)
        data = data * 2. - 1.
        return data

    def inverse_transform(self, data):
        data = (data + 1.) / 2.
        data = 1. * data * (self._max - self._min) + self._min
        return data

# Define cubic spline interpolation function
def cubic_spline_interpolation(input_tensor, num_interp_points):
    """
    :param input_tensor: Input tensor with shape (batch_size, num_points, xy_coords)
    :param num_interp_points: Number of interpolation points between each pair of points
    :return: Interpolated tensor
    """
    batch_size, num_points, _ = input_tensor.shape
    device = input_tensor.device
    
        # Convert the input tensor to a NumPy array
    if input_tensor.requires_grad:
        input_tensor = input_tensor.detach()
    input_np = input_tensor.cpu().numpy()
    
    # Initialize the output list
    interp_results = []
    
    for sample in input_np:
        x = sample[:, 0]  # Extract x coordinates
        y = sample[:, 1]  # Extract y coordinates
        
        # Create cubic spline interpolation objects
        cs_x = CubicSpline(np.arange(num_points), x)  # Interpolate x coordinates
        cs_y = CubicSpline(np.arange(num_points), y)  # Interpolate y coordinates
        
        # Generate interpolation points
        if num_interp_points == 1:
            interp_indices = np.linspace(0, num_points - 1, 2 * num_points - 1)
        interp_x = cs_x(interp_indices)
        interp_y = cs_y(interp_indices)
        
        # Merge interpolation results
        interp_sample = np.stack([interp_x, interp_y], axis=-1)
        interp_results.append(interp_sample)
    
    # Convert the results back to a PyTorch tensor
    interp_tensor = torch.tensor(np.array(interp_results), dtype=torch.float64, device=device)
    return interp_tensor
