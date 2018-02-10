hidden_state_size = 5;
batch_size = 32;

# This is referred above as f(u).
class nn_MSECriterion:
    def forward(self, predictions, labels):
        return np.sum(np.square(predictions - labels))
        
    def backward(self, predictions, labels):
        num_samples = labels.shape[0]
        return num_samples * 2 * (predictions - labels)

# This is referred above as g(v).
class nn_Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x, gradOutput):
        # It is usually a good idea to use gv from the forward pass and not recompute it again here.
        gv = 1 / (1 + np.exp(-x))  
        return np.multiply(np.multiply(gv, (1 - gv)), gradOutput)

# This is referred above as h(W, b)
class nn_Linear:
    def __init__(self, input_dim, output_dim):
        # Initialized with random numbers from a gaussian N(0, 0.001)
        self.weight = np.matlib.randn(input_dim, output_dim) * 0.01
        self.bias = np.matlib.randn((1, output_dim)) * 0.01
        self.gradWeight = np.zeros_like(self.weight)
        self.gradBias = np.zeros_like(self.bias)
    
    #New function to reinitialize gradweights to zero in each batch
    def reinitGrads(self):
        self.gradWeight = np.zeros_like(self.weight)
        self.gradBias = np.zeros_like(self.bias)
    
    def forward(self, x):
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, x, gradOutput):
        # dL/dw = dh/dw * dL/dv
        self.gradWeight = self.gradWeight + np.dot(x.T, gradOutput)
        # dL/db = dh/db * dL/dv
        self.gradBias = self.gradBias + np.copy(gradOutput)
        # return dL/dx = dh/dx * dL/dv
        return np.dot(gradOutput, self.weight.T)
    
    def getParameters(self):
        params = [self.weight, self.bias]
        gradParams = [self.gradWeight, self.gradBias]
        return params, gradParams
    

learningRate = 0.001

model = {}
model['linear1'] = nn_Linear(4, hidden_state_size)
model['linear2'] = nn_Linear(hidden_state_size, 3)
model['sigmoid'] = nn_Sigmoid()
model['loss'] = nn_MSECriterion()

for epoch in range(0, 300):
    loss = 0
    for i in range(0, int(dataset_size/batch_size)):
        
        model['linear1'].reinitGrads()
        model['linear2'].reinitGrads()
        
        for k in range(0,batch_size):
            xi = x[i*batch_size+k:i*batch_size+k+1, :]
            yi = y[i*batch_size+k:i*batch_size+k+1, :]

            # Forward.
            a0 = model['linear1'].forward(xi)
            a1 = model['sigmoid'].forward(a0)
            a2 = model['linear2'].forward(a1)
            a3 = model['sigmoid'].forward(a2)
            loss += model['loss'].forward(a3, yi)

            # Backward.
            da3 = model['loss'].backward(a3, yi)
            da2 = model['sigmoid'].backward(a2, da3)
            da1 = model['linear2'].backward(a1, da2)
            da0 = model['sigmoid'].backward(a0, da1)
            dxi = model['linear1'].backward(xi, da0)
            
            
        #update weights per batch 
        model['linear2'].weight = model['linear2'].weight - learningRate * model['linear2'].gradWeight
        model['linear2'].bias = model['linear2'].bias - learningRate * model['linear2'].gradBias
        model['linear1'].weight = model['linear1'].weight - learningRate * model['linear1'].gradWeight
        model['linear1'].bias = model['linear1'].bias - learningRate * model['linear1'].gradBias
    
    if epoch % 10 == 0: print('epoch[%d] = %.8f' % (epoch, loss / dataset_size))