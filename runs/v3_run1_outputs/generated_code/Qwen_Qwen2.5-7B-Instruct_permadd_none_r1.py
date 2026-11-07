def build_weights():
    import math
    
    def get_element_index(s):
        return VAL.index(s)
    
    def set_weight(w1, w2, out_symbol, in_symbol_a, in_symbol_b):
        a_index = get_element_index(in_symbol_a)
        b_index = get_element_index(in_symbol_b)
        c_index = get_element_index(out_symbol)
        
        # Calculate the weight based on the element indices
        w1[out_symbol, a_index * 2 + b_index] = 1.0
        
        # For the second layer, we need to ensure the activation matches the desired output
        # Weights are set such that the activation pattern corresponds to the correct output symbol
        # Here, we use the fact that the activation is element-wise squared to map to the correct output
        # The specific values are derived from the requirement that the output should be correct for all inputs
        w2[out_symbol, a_index * 256 + b_index] = math.sqrt(1 / 113)
    
    # Initialize weight matrices
    W1 = np.zeros((113, 226))
    W2 = np.zeros((113, 256))
    
    # Fill in W1 and W2
    for a in range(113):
        for b in range(113):
            out_symbol = (a + b) % 113
            set_weight(W1, W2, out_symbol, a, b)
    
    return {"W1": W1, "W2": W2}