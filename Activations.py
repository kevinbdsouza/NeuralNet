# Rectified linear unit
class nn_ReLU:
    def forward(self, x):
        # Forward pass.
        return np.maximum(0,x)
    
    def backward(self, x, gradOutput):
        # Backward pass
        if (x<0):
            return 0
        
        if (x>0):
            return gradOutput 
    
        if (x==0):
            return np.multiply(0.5,gradOutput)
        
# Hyperbolic tangent.
class nn_Tanh:
    def forward(self, x):
        # Forward pass.
        return np.exp(x) - np.exp(-x)/ np.exp(x) + np.exp(-x)
    
    def backward(self, x, gradOutput):
        # Backward pass
        return np.multiply(1 - np.square(np.exp(x) - np.exp(-x)/ np.exp(x) + np.exp(-x)),gradOutput)
