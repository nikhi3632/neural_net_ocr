import numpy as np
np.random.seed(0)

# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size, out_size, params, name = ''):
    # Xavier initialization uniform random distribution in [-a, a]
    # variance = (a-(-a))^2/12 = 2/(in_size + out_size)
    a = np.sqrt(6 / (in_size+out_size))
    W = np.random.uniform(-1*a, a, (in_size, out_size))
    b = np.zeros(out_size)
    params['W' + name] = W
    params['b' + name] = b

# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def forward(X, params, name = '', activation = sigmoid):
    """
    Do a forward pass
    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]
    pre_act = np.dot(X, W) + b # XW + b
    post_act = activation(pre_act) # f(XW + b)
    # store the pre-activation and post-activation values to use in backprop
    params['cache_' + name] = (X, pre_act, post_act)
    return post_act

# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# compute total loss and accuracy
# y is size [examples, classes]
# probs is size [examples, classes]
def compute_loss_and_acc(y, probs):
    examples = y.shape[0]
    # Calculate cross-entropy loss
    epsilon = 1e-10
    loss = -np.sum(y * np.log(probs+ epsilon))
    # Calculate accuracy
    acc = np.sum(np.argmax(y, axis=1) == np.argmax(probs, axis=1)) / examples
    return loss, acc

# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta, params, name = '', activation_deriv = sigmoid_deriv):
    """
    Do a backwards pass
    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    # everything needed for this layer
    W = params['W' + name]
    X, _ , post_act = params['cache_' + name]
    # do the derivative through activation first
    # then compute the derivative W, b, and X
    act_deriv = activation_deriv(post_act) * delta # post_act = f(XW + b)
    grad_W = X.T @ act_deriv                       # in_size x out_size, d/dW(f(XW + b))
    grad_b = np.ones(len(act_deriv)) @ act_deriv   # 1 x out_size, d/db(f(XW + b))
    grad_X = act_deriv @ W.T                       # d/dX(f(XW + b))
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x, y, batch_size):
    batches = []
    nx = x.shape[0]
    # Calculate the number of batches
    num_batches = int(np.ceil(nx / batch_size))
    for _ in range(num_batches):
        # Randomly sample indices for the current batch
        idx = np.random.choice(nx, size=batch_size, replace=False)
        # Extract the batch from the input data
        bx = x[idx, :]
        by = y[idx, :]
        # Append the batch to the list of batches
        batches.append((bx, by))
    return batches
