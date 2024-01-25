# Forward propagation, Backward propagation, Training Loop and Numerical Gradient Checker. 

from neural_net import initialize_weights, forward, backwards, \
    sigmoid, sigmoid_deriv, softmax, compute_loss_and_acc, get_random_batches
from util import linear_deriv
import copy
import warnings
warnings.filterwarnings('ignore')
import numpy as np
np.random.seed(42)

# Generate fake data
g0 = np.random.multivariate_normal([3.6, 40],[[0.05, 0],[0, 10]], 10)
g1 = np.random.multivariate_normal([3.9, 10],[[0.01, 0],[0, 5]], 10)
g2 = np.random.multivariate_normal([3.4, 30],[[0.25, 0],[0, 5]], 10)
g3 = np.random.multivariate_normal([2.0, 10],[[0.5, 0],[0, 10]], 10)
x = np.vstack([g0, g1, g2, g3])
# do XW + B that implies that the data is N x D

# create labels
y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])
# turn to one_hot
y = np.zeros((y_idx.shape[0], y_idx.max() + 1))
y[np.arange(y_idx.shape[0]), y_idx] = 1

# parameters in a dictionary
params = {}

# initialize a layer
initialize_weights(2, 25, params, 'layer1')
initialize_weights(25, 4, params, 'output')
assert(params['W' + 'layer1'].shape == (2,25))
assert(params['b' + 'layer1'].shape == (25,))

#expect 0, [0.05 to 0.12]
print("{}, {:.2f}".format(params['blayer1'].sum(), params['W' + 'layer1'].std()**2))
print("{}, {:.2f}".format(params['boutput'].sum(), params['W' + 'output'].std()**2))

# implement sigmoid
test = sigmoid(np.array([-1000, 1000]))
print('should be zero and one\t', test.min(), test.max())
# implement forward
h1 = forward(x, params, 'layer1')
# print(h1.shape)

# implement softmax
probs = forward(h1, params, 'output', softmax)
# make sure to understand these values!
# positive, ~1, ~1, (40,4)
# print(probs.min(), min(probs.sum(1)), max(probs.sum(1)), probs.shape)

# implement compute_loss_and_acc
loss, acc = compute_loss_and_acc(y, probs)
# if it is not, check softmax!
# print("{}, {:.2f}".format(loss, acc))

# the derivative of cross-entropy(softmax(x)) is probs - 1[correct actions]
delta1 = probs
delta1[np.arange(probs.shape[0]), y_idx] -= 1

# Already did derivative through softmax
# so pass in a linear_deriv, which is just a vector of ones
# to make this a no-op
delta2 = backwards(delta1, params, 'output', linear_deriv)
# Implement backwards!
backwards(delta2, params, 'layer1', sigmoid_deriv)

# W and b should match their gradients sizes
for k, v in sorted(list(params.items())):
    if 'grad' in k:
        name = k.split('_')[1]
        print(name, v.shape, params[name].shape)

batches = get_random_batches(x, y, 5)
# print batch sizes
# print([batch[0].shape[0] for batch in batches])
batch_num = len(batches)

def compute_gradient(params, name, eta):
    params['W' + name] -= eta*params['grad_W' + name]
    params['b' + name] -= eta*params['grad_b' + name]

# TRAINING LOOP
max_iters = 500
learning_rate = 1e-3

for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for xb, yb in batches:
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        # loss
        loss, acc = compute_loss_and_acc(yb, probs)
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss
        avg_acc += acc/batch_num
        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)
        # apply gradient
        compute_gradient(params, 'output', learning_rate)
        compute_gradient(params, 'layer1', learning_rate)
        
    if itr % 100 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, avg_acc))

# Do a forward & backward pass of the dataset here to get params populated with the gradient expected
xb, yb = batches[0]
out = forward(xb, params, "layer1", sigmoid)
probs = forward(out, params, "output", softmax)
loss, acc = compute_loss_and_acc(yb, probs)
delta = probs - yb
delta1 = backwards(delta, params, "output", linear_deriv)
delta2 = backwards(delta1, params, "layer1", sigmoid_deriv)

# save the old params and the gradients that are just computed
params_orig = copy.deepcopy(params)

# Gradient obtained from Backpropagation
h1 = forward(x, params_orig, 'layer1', sigmoid)
probs = forward(h1, params_orig, 'output', softmax)
delta1 = probs - y
delta2 = backwards(delta1, params_orig, 'output', linear_deriv)
backwards(delta2, params_orig, 'layer1', sigmoid_deriv)

def forward_pass_loss(params):
    h1 = forward(x, params, 'layer1', sigmoid)
    probs = forward(h1, params, 'output', softmax)
    loss, _ = compute_loss_and_acc(y, probs)
    return loss

# Get the same result with numerical gradients instead of using the analytical gradients computed from the chain rule.
# Add epsilon offset to each element in the weights, and compute the numerical gradient of the loss with central differences. 
# Central differences is just (f(x + eps)−f(x - eps))/(2*eps).
# This needs to be done for each scalar dimension in all of the weights independently.
eps = 1e-6
for k, v in params.items():
    if '_' in k: 
        continue
    # There is a real parameter!
    # for each value inside the parameter
    #   add epsilon
    #   run the network
    #   get the loss
    #   compute derivative with central diffs
    #   store that inside params
    if len(params[k].shape) > 1: # Weights
        for r in range(params[k].shape[0]):
            for c in range(params[k].shape[1]):
                value = params[k][r, c].copy()
                params[k][r, c] = value + eps
                loss1 = forward_pass_loss(params)
                params[k][r, c] = value - eps
                loss2 = forward_pass_loss(params)
                params[k][r, c] = value
                # Compute numerical gradient
                params['grad_' + k][r, c] = (loss1 - loss2) / (2*eps)
    else:  # Bias
        for r in range(params[k].shape[0]):
            value = params[k][r].copy()
            params[k][r] = value + eps
            loss1 = forward_pass_loss(params)
            params[k][r] = value - eps
            loss2 = forward_pass_loss(params)
            params[k][r] = value
            # Compute numerical gradient
            params['grad_' + k][r] = (loss1 - loss2) / (2*eps)

total_error = 0
for k in params.keys():
    if 'grad_' in k:
        # relative error
        err = np.abs(params[k] - params_orig[k])/np.maximum(np.abs(params[k]), np.abs(params_orig[k]))
        err = err.sum()
        print('{} {:.2e}'.format(k, err))
        total_error += err
# should be less than 1e-4
assert total_error < 1e-4   
print('total {:.2e}'.format(total_error))
