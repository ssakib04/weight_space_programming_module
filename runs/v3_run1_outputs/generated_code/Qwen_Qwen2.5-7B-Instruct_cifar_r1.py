import numpy as np

def build_weights():
    # Initialize weights and biases
    conv1_weight = np.zeros((32, 3, 5, 5))
    conv1_bias = np.zeros(32)
    conv2_weight = np.zeros((64, 32, 3, 3))
    conv2_bias = np.zeros(64)
    fc_weight = np.zeros((10, 1024))
    fc_bias = np.zeros(10)
    
    # Constructing the weights
    
    # Conv1 Filters (oriented edges, colors, simple shapes)
    for i in range(32):
        if i < 8:
            # Oriented Edges
            conv1_weight[i, :, 2, 2] = 1 / math.sqrt(5)
        elif i < 16:
            # Color-opponent (Green-Yellow)
            conv1_weight[i, 0, 2, 2] = -0.5
            conv1_weight[i, 1, 2, 2] = 0.5
        elif i < 24:
            # Color-opponent (Blue-Red)
            conv1_weight[i, 0, 2, 2] = 0.5
            conv1_weight[i, 2, 2, 2] = -0.5
        else:
            # Center-Surround
            conv1_weight[i, :, 2, 2] = 1 / math.sqrt(11)
            conv1_weight[i, :, 1, 1] = -1 / math.sqrt(11)
    
    # Conv2 Filters (Texture and Shape Detectors)
    for i in range(64):
        if i < 32:
            # Blob Detectors
            conv2_weight[i, :, 1, 1] = 1 / math.sqrt(9)
        else:
            # Combining textures and simple shapes
            conv2_weight[i, :, 1, 1] = 1 / math.sqrt(9)
            conv2_weight[i, :, 1, 2] = -1 / math.sqrt(9)
            conv2_weight[i, :, 2, 1] = -1 / math.sqrt(9)
            conv2_weight[i, :, 2, 2] = 1 / math.sqrt(9)
    
    # FC Weight (Prior Knowledge)
    fc_weight[0, :256] = 1 / math.sqrt(256)  # Airplane
    fc_weight[1, 256:512] = 1 / math.sqrt(256)  # Automobile
    fc_weight[2, 512:768] = 1 / math.sqrt(256)  # Bird
    fc_weight[3, 768:1024] = 1 / math.sqrt(256)  # Cat
    fc_weight[4, :256] = 1 / math.sqrt(256)  # Deer
    fc_weight[5, 256:512] = 1 / math.sqrt(256)  # Dog
    fc_weight[6, 512:768] = 1 / math.sqrt(256)  # Frog
    fc_weight[7, 768:1024] = 1 / math.sqrt(256)  # Horse
    fc_weight[8, :256] = 1 / math.sqrt(256)  # Ship
    fc_weight[9, 256:512] = 1 / math.sqrt(256)  # Truck
    
    # Biases (Assume some bias for each class to ensure non-zero output)
    for i in range(10):
        fc_bias[i] = 0.1
    
    return {
        "conv1.weight": conv1_weight,
        "conv1.bias": conv1_bias,
        "conv2.weight": conv2_weight,
        "conv2.bias": conv2_bias,
        "fc.weight": fc_weight,
        "fc.bias": fc_bias
    }