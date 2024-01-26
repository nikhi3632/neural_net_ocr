import numpy as np
import scipy.io
from neural_net import initialize_weights, forward, backwards, softmax, \
    sigmoid, sigmoid_deriv, get_random_batches, compute_loss_and_acc
from util import linear_deriv
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.axes_grid1 import ImageGrid
import string
import shutil
import os

ARTIFACTS_DIR = os.getcwd() + "/artifacts"

def create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

create_dir(ARTIFACTS_DIR)

train_data = scipy.io.loadmat('../data/nist/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

# pick parameters
max_iters = 50
batch_size = 32
learning_rate = 1e-2
hidden_size = 64

batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
n_input_layer = train_x.shape[1]
n_output_layer = train_y.shape[1]
initialize_weights(n_input_layer, hidden_size, params, 'layer1')
initialize_weights(hidden_size, n_output_layer, params, 'output')

def visualize_weights(W, save_path):
    assert W.shape == (32 * 32, 64)
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(8, 8))
    for i in range(64):
        grid[i].imshow(W[:, i].reshape(32, 32))
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

visualize_weights(params['W' + 'layer1'], ARTIFACTS_DIR + '/initial_weights.png')

def compute_gradient(params, name, eta):
    params['W' + name] -= eta*params['grad_W' + name]
    params['b' + name] -= eta*params['grad_b' + name]

n_examples = train_x.shape[0]
x = np.arange(max_iters)
train_acc_list = []
valid_acc_list = []
train_loss_list = []
valid_loss_list = []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # forward
        out = forward(xb, params, "layer1", sigmoid)
        probs = forward(out, params, "output", softmax)
        # loss be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc
        # backward
        delta = probs - yb
        delta = backwards(delta, params, "output", linear_deriv)
        delta = backwards(delta, params, "layer1", sigmoid_deriv)
        # apply gradient
        compute_gradient(params, 'output', learning_rate)
        compute_gradient(params, 'layer1', learning_rate)
    
    total_acc /= batch_num
    total_loss /= n_examples   

    out = forward(valid_x, params, "layer1", sigmoid)
    probs = forward(out, params, "output", softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
    valid_loss /= valid_x.shape[0]

    train_acc_list.append(total_acc)
    valid_acc_list.append(valid_acc)
    train_loss_list.append(total_loss)
    valid_loss_list.append(valid_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, total_acc))

plt.plot(x, train_acc_list, linewidth = 3, label = "Training Accuracy")
plt.plot(x, valid_acc_list, linewidth = 3, label = "Validation Accuracy")
plt.legend()
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(ARTIFACTS_DIR + "/accuracy_plot.png")
plt.close()

# Plotting Loss
plt.plot(x, train_loss_list, linewidth = 3, label = "Training Loss")
plt.plot(x, valid_loss_list, linewidth = 3, label = "Validation Loss")
plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(ARTIFACTS_DIR + "/loss_plot.png")
plt.close()

# run on validation set and report accuracy!
valid_acc = None
valid_out = forward(valid_x, params, "layer1", sigmoid)
valid_probs = forward(valid_out, params, "output", softmax)
_, valid_acc = compute_loss_and_acc(valid_y, valid_probs)

print('Validation accuracy: ', valid_acc)

test_acc = None
test_out = forward(test_x, params, "layer1", sigmoid)
test_probs = forward(test_out, params, "output", softmax)
_, test_acc = compute_loss_and_acc(test_y, test_probs)

print('Test accuracy: ', test_acc)

saved_params = {k:v for k, v in params.items() if '_' not in k}
with open(ARTIFACTS_DIR + '/model_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

visualize_weights(params['W' + 'layer1'], ARTIFACTS_DIR + '/learned_weights.png')

confusion_matrix = np.zeros((train_y.shape[1], train_y.shape[1]))
valid_pred_y = np.argmax(valid_probs, axis = 1)
for i in range(valid_pred_y.shape[0]):
    pred = valid_pred_y[i]
    label = np.argmax(valid_y[i])
    confusion_matrix[label][pred] += 1

plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.savefig(ARTIFACTS_DIR + '/confusion_matrix')
plt.close()
