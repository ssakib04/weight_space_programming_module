import numpy as np
import math

def build_weights():
    def gabor_filter(ksize, sigma, theta, lamda, psi, gamma):
        """Generate a single Gabor filter."""
        sigma_x = sigma
        sigma_y = sigma / gamma
        
        # Create a grid of coordinates
        x = np.linspace(-ksize//2, ksize//2, ksize)
        y = np.linspace(-ksize//2, ksize//2, ksize)
        x, y = np.meshgrid(x, y)
        
        # Rotate the grid
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)
        
        # Create the Gabor filter
        gbreal = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * np.cos(2 * math.pi / lamda * x_theta + psi)
        return gbreal

    # Initialize weights
    conv1_weight = np.zeros((32, 3, 5, 5))
    conv1_bias = np.zeros(32)
    conv2_weight = np.zeros((64, 32, 3, 3))
    conv2_bias = np.zeros(64)
    fc_weight = np.zeros((10, 1024))
    fc_bias = np.zeros(10)

    # Define some parameters for Gabor filters
    sigma = 1.0
    psi = 0.0
    gamma = 1.0
    lamdas = [3.0, 4.0, 6.0, 8.0]
    thetas = [0, math.pi/4, math.pi/2, 3*math.pi/4]

    # Create Gabor filters for conv1
    filter_index = 0
    for theta in thetas:
        for lamda in lamdas:
            if filter_index >= 32:
                break
            filter = gabor_filter(5, sigma, theta, lamda, psi, gamma)
            conv1_weight[filter_index, :, :, :] = filter
            filter_index += 1

    # Define some simple texture filters for conv2
    conv2_weight[0] = np.array([[[[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                                 [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                                 [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]],
                                [[[1, 0, -1], [1, 0, -1], [1, 0, -1]],
                                 [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
                                 [[1, 0, -1], [1, 0, -1], [1, 0, -1]]],
                                [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                 [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]])
    conv2_weight[1] = np.array([[[[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                                 [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                                 [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]],
                                [[[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                                 [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                                 [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]],
                                [[[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                                 [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                                 [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]])

    # Simple classifier head based on prior knowledge
    fc_weight[0, :256] = 1  # Airplane and Ship
    fc_weight[1, 256:512] = 1  # Automobile
    fc_weight[2, 512:768] = 1  # Bird
    fc_weight[3, 768:] = 1  # Cat
    fc_weight[4, :256] = -1  # Deer
    fc_weight[4, 256:512] = -1  # Deer
    fc_weight[5, 512:768] = -1  # Dog
    fc_weight[6, 768:] = -1  # Frog
    fc_weight[7, :256] = -1  # Horse
    fc_weight[7, 256:512] = -1  # Horse
    fc_weight[8, 512:768] = -1  # Ship
    fc_weight[9, 768:] = -1  # Truck

    return {
        "conv1.weight": conv1_weight,
        "conv1.bias": conv1_bias,
        "conv2.weight": conv2_weight,
        "conv2.bias": conv2_bias,
        "fc.weight": fc_weight,
        "fc.bias": fc_bias
    }