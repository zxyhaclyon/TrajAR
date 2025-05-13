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

# 定义三次样条插值函数
def cubic_spline_interpolation(input_tensor, num_interp_points):
    """
    :param input_tensor: 输入张量，形状为 (batch_size, num_points, xy_coords)
    :param num_interp_points: 每两个点之间的插值点数量
    :return: 插值后的张量
    """
    batch_size, num_points, _ = input_tensor.shape
    device = input_tensor.device
    
    # 将输入张量转换为 NumPy 数组（因为 scipy 不支持 PyTorch 张量）
    if input_tensor.requires_grad:
        input_tensor = input_tensor.detach()
    input_np = input_tensor.cpu().numpy()
    
    # 初始化输出列表
    interp_results = []
    
    for sample in input_np:
        x = sample[:, 0]  # 提取 x 坐标
        y = sample[:, 1]  # 提取 y 坐标
        
        # 创建三次样条插值对象
        cs_x = CubicSpline(np.arange(num_points), x)  # 对 x 坐标插值
        cs_y = CubicSpline(np.arange(num_points), y)  # 对 y 坐标插值
        
        # 生成插值点
        if num_interp_points == 1:
            interp_indices = np.linspace(0, num_points - 1, 2 * num_points - 1)
        interp_x = cs_x(interp_indices)
        interp_y = cs_y(interp_indices)
        
        # 合并插值结果
        interp_sample = np.stack([interp_x, interp_y], axis=-1)
        interp_results.append(interp_sample)
    
    # 将结果转换为 PyTorch 张量
    interp_tensor = torch.tensor(np.array(interp_results), dtype=torch.float64, device=device)
    return interp_tensor