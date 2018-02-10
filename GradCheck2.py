hidden_state_size = 5;
# Your code goes here.
learningRate = 0.1

x = np.array([[2.34, 3.8, 34.44, 5.33]])
y = np.array([[3.2, 4.2, 5.3]])

model = {}
model['linear1'] = nn_Linear(4, hidden_state_size)
model['linear2'] = nn_Linear(hidden_state_size, 3)
model['sigmoid'] = nn_Sigmoid()
model['loss'] = nn_MSECriterion()

a0 = model['linear1'].forward(x)
a1 = model['sigmoid'].forward(a0)
a2 = model['linear2'].forward(a1)
a3 = model['sigmoid'].forward(a2)
#loss += model['loss'].forward(a3, y)

# Backward.
da3 = model['loss'].backward(a3, y)
da2 = model['sigmoid'].backward(a2, da3)
da1 = model['linear2'].backward(a1, da2)

da0 = model['sigmoid'].backward(a0, da1)
dxi = model['linear1'].backward(x, da0)


gradWeight1 = model['linear1'].gradWeight
gradWeight2 = model['linear2'].gradWeight

approxGradWeight1 = np.zeros_like(model['linear1'].weight)
approxGradWeight2 = np.zeros_like(model['linear2'].weight)


epsilon = 0.001

#checking gradients in the first layer (have to repass the input as the weights have changed now) 
for i in range(0, model['linear1'].weight.shape[0]):
    for j in range(0, model['linear1'].weight.shape[1]):
        # Compute f(w)
        fw = model['loss'].forward(a3, y) # Loss function.
        # Compute f(w + eps)
        shifted_weight1 = np.copy(model['linear1'].weight)
        shifted_weight1[i, j] = shifted_weight1[i, j] + epsilon
        shifted_linear1 = nn_Linear(4, hidden_state_size)
        shifted_linear1.bias = model['linear1'].bias
        shifted_linear1.weight = shifted_weight1
        fw_epsilon = model['loss'].forward(model['sigmoid'].forward(model['linear2'].forward(
            model['sigmoid'].forward(shifted_linear1.forward(x)))), y) # Loss function
        # Compute (f(w + eps) - f(w)) / eps
        approxGradWeight1[i, j] = (fw_epsilon - fw) / epsilon

print('gradWeight: ' + str(gradWeight1))
print('\napproxGradWeight: ' + str(approxGradWeight1))

#checking gradients in the hidden layer (have to repass only to the second layer)
for i in range(0, model['linear2'].weight.shape[0]):
    for j in range(0, model['linear2'].weight.shape[1]):
        # Compute f(w)
        fw = model['loss'].forward(a3, y) # Loss function.
        # Compute f(w + eps)
        shifted_weight2 = np.copy(model['linear2'].weight)
        shifted_weight2[i, j] = shifted_weight2[i, j] + epsilon
        shifted_linear2 = nn_Linear(hidden_state_size, 3)
        shifted_linear2.bias = model['linear2'].bias
        shifted_linear2.weight = shifted_weight2
        fw_epsilon = model['loss'].forward(model['sigmoid'].forward(shifted_linear2.forward(a1)), y) # Loss function
        # Compute (f(w + eps) - f(w)) / eps
        approxGradWeight2[i, j] = (fw_epsilon - fw) / epsilon
        
# These two outputs should be similar up to some precision.
print('gradWeight: ' + str(gradWeight2))
print('\napproxGradWeight: ' + str(approxGradWeight2))

