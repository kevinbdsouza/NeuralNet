# Build a two-layer neural network (so one hidden layer) with sigmoid activations 
# and MSE loss. The hidden_state_dimensionality should be set to 1 using the variable
# below.
hidden_state_size = 1; 

# Define the 2-layer network here (fill in your code)
model = {}
model['linear1'] = nn_Linear(2, hidden_state_size)
model['linear2'] = nn_Linear(hidden_state_size, 1)
model['sigmoid'] = nn_Sigmoid()
model['loss'] = nn_MSECriterion()

# Optimize the parameters of the neural network using stochastic gradient descent
# using the following parameters

learningRate = 0.00001
numberEpochs = 300

for epoch in range(0, numberEpochs):
    loss = 0
    for i in range(0, Y.size):
        xi = X.T[i:i+1, :]
        yi = Y.T[i:i+1, :]
        
        # Forward pass (fill in your code)
        a0 = model['linear1'].forward(xi)
        a1 = model['sigmoid'].forward(a0)
        a2 = model['linear2'].forward(a1)
        a3 = model['sigmoid'].forward(a2)
        loss += model['loss'].forward(a3, yi)    
    
        # Backward pass (fill in your code)
        da3 = model['loss'].backward(a3, yi)
        da2 = model['sigmoid'].backward(a2, da3)
        da1 = model['linear2'].backward(a1, da2)    
        da0 = model['sigmoid'].backward(a0, da1)
        dxi = model['linear1'].backward(xi, da0)
        
        # Update gradients (fill in your code)
        model['linear2'].weight = model['linear2'].weight - learningRate * model['linear2'].gradWeight
        model['linear2'].bias = model['linear2'].bias - learningRate * model['linear2'].gradBias
        model['linear1'].weight = model['linear1'].weight - learningRate * model['linear1'].gradWeight
        model['linear1'].bias = model['linear1'].bias - learningRate * model['linear1'].gradBias

    if epoch % 10 == 0: print('epoch[%d] = %.8f' % (epoch, loss / dataset_size))


%matplotlib inline

classEstimate = np.zeros((400,1), dtype='uint8')

for i in range(0, 400):  
    xi = X.T[i:i+1, :]
    
    # Forward pass (fill in your code)
    a0 = model['linear1'].forward(xi)
    a1 = model['sigmoid'].forward(a0)
    a2 = model['linear2'].forward(a1)
    y_hat = model['sigmoid'].forward(a2)   
        
    classEstimate[i,0] = (y_hat > 0.5)

plt.scatter(X[0, :], X[1, :], c=classEstimate[:,0], s=40, cmap=plt.cm.Spectral);