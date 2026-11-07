import numpy as np
import math

def build_weights():
    # Initialize weights and biases for conv1
    conv1_weight = np.zeros((32, 3, 5, 5))
    conv1_bias = np.zeros(32)
    
    # Initialize weights and biases for conv2
    conv2_weight = np.zeros((64, 32, 3, 3))
    conv2_bias = np.zeros(64)
    
    # Initialize weights and biases for fc
    fc_weight = np.zeros((10, 1024))
    fc_bias = np.zeros(10)
    
    # Define some constants
    pi = math.pi
    
    # Create Gabor filters for conv1
    for i in range(8):
        sigma = 1.0
        theta = i * (pi / 8)
        lamda = 2.0
        gamma = 0.5
        phi = 0.0
        
        for c in range(3):
            for x in range(5):
                for y in range(5):
                    x0 = 2 * x - 4
                    y0 = 2 * y - 4
                    x_theta = x0 * math.cos(theta) + y0 * math.sin(theta)
                    y_theta = -x0 * math.sin(theta) + y0 * math.cos(theta)
                    gabor_real = math.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * math.cos(2 * pi * x_theta / lamda + phi)
                    conv1_weight[i, c, x, y] = gabor_real
    
    # Create color-opponent filters for conv1
    conv1_weight[8, 0, :, :] = 1  # Red
    conv1_weight[9, 1, :, :] = 1  # Green
    conv1_weight[10, 2, :, :] = 1  # Blue
    conv1_weight[11, 0, :, :] = -1  # Red-Green opponent
    conv1_weight[11, 1, :, :] = 1
    conv1_weight[12, 1, :, :] = -1  # Green-Blue opponent
    conv1_weight[12, 2, :, :] = 1
    conv1_weight[13, 0, :, :] = -1  # Red-Blue opponent
    conv1_weight[13, 2, :, :] = 1
    
    # Create center-surround filters for conv1
    conv1_weight[14, :, 2, 2] = 1
    conv1_weight[14, :, 1, 1] = -1
    conv1_weight[14, :, 3, 3] = -1
    conv1_weight[15, :, 2, 2] = -1
    conv1_weight[15, :, 1, 1] = 1
    conv1_weight[15, :, 3, 3] = 1
    
    # Create blob detectors for conv1
    for i in range(16, 32):
        sigma = 1.0
        for c in range(3):
            for x in range(5):
                for y in range(5):
                    x0 = 2 * x - 4
                    y0 = 2 * y - 4
                    blob_value = math.exp(-(x0**2 + y0**2) / (2 * sigma**2))
                    conv1_weight[i, c, x, y] = blob_value
    
    # Create texture detectors for conv2 (simple combinations of conv1 features)
    for i in range(32):
        for j in range(32):
            if i % 2 == 0 and j % 2 == 0:
                conv2_weight[i, j, :, :] = 1
            elif i % 2 == 1 and j % 2 == 1:
                conv2_weight[i, j, :, :] = -1
    
    # Create classifier head for fc (prior knowledge of classes)
    # Airplane: blue/uniform top regions
    fc_weight[0, 8*64+0:8*64+16] = 1
    fc_weight[0, 8*64+16:8*64+32] = 1
    fc_weight[0, 8*64+32:8*64+48] = 1
    fc_weight[0, 8*64+48:8*64+64] = 1
    
    # Ship: blue/uniform top regions
    fc_weight[1, 8*64+0:8*64+16] = 1
    fc_weight[1, 8*64+16:8*64+32] = 1
    fc_weight[1, 8*64+32:8*64+48] = 1
    fc_weight[1, 8*64+48:8*64+64] = 1
    
    # Frog: green/brown textures
    fc_weight[2, 9*64+0:9*64+16] = 1
    fc_weight[2, 9*64+16:9*64+32] = 1
    fc_weight[2, 9*64+32:9*64+48] = 1
    fc_weight[2, 9*64+48:9*64+64] = 1
    
    # Deer: green/brown textures
    fc_weight[3, 9*64+0:9*64+16] = 1
    fc_weight[3, 9*64+16:9*64+32] = 1
    fc_weight[3, 9*64+32:9*64+48] = 1
    fc_weight[3, 9*64+48:9*64+64] = 1
    
    # Dog: general detection
    fc_weight[4, :] = 1
    
    # Horse: general detection
    fc_weight[5, :] = 1
    
    # Bird: general detection
    fc_weight[6, :] = 1
    
    # Cat: general detection
    fc_weight[7, :] = 1
    
    # Truck: general detection
    fc_weight[8, :] = 1
    
    # Automobile: general detection
    fc_weight[9, :] = 1
    
    return {
        "conv1.weight": conv1_weight,
        "conv1.bias": conv1_bias,
        "conv2.weight": conv2_weight,
        "conv2.bias": conv2_bias,
        "fc.weight": fc_weight,
        "fc.bias": fc_bias
    }