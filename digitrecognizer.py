import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
filepath = 'C:/elaine/projects/digitrecognizer/digit-recognizer/train.csv'
data = pd.read_csv(filepath)

##########################
### SPLITTING THE DATA ###
##########################

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

#defining weights and biases
def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

##########################
### TRAINING FUNCTIONS ###
##########################

#defining the activation functions
def relu(z):
    """
    ReLU activation function. Returns z when z > 0 and 0 otherwise.
    """
    return np.maximum(z, 0)

def derivative_relu(z):
    """
    The derivative of the ReLU function. Returns 1 when z > 0 and 0 otherwise.
    """
    return z > 0

def softmax(z):
    """
    Softmax function.
    """
    ret = np.exp(z) / sum(np.exp(z))
    return ret

def onehot(y):
    """
    One-hot encoding the y.
    """
    onehot_y = np.zeros((y.size, y.max() + 1))
    onehot_y[np.arange(y.size), y] = 1
    onehot_y = onehot_y.T
    return onehot_y

def forward_prop(w1, b1, w2, b2, x):
    """
    Forward propagation.
    """
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2
    
def backward_prop(z1, a1, z2, a2, w1, w2, x, y):
    """
    Backward propagation.
    """
    onehot_y = onehot(y)
    dz2 = a2 - onehot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    
    dz1 = w2.T.dot(dz2) * derivative_relu(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)
    
    return dw2, db2, dw1, db1

def update_params(dw2, db2, dw1, db1, w1, b1, w2, b2, alpha):
    """
    Updating the parameters after each run through of forward and back
    propagation.
    """
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2
    
def get_predictions(a2):
    """
    Helper function to see the predictions while it trains.
    """
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    """
    Helper function to see the accuracies while it trains.
    """
    print(predictions, y)
    return np.sum(predictions == y)/y.size

def gradient_descent(x, y, iterations, alpha):
    """
    The gradient descent model, as well as the mid-training prints.
    Cycles through forward propagation, backward propagation, and
    minimizing losses/updating parameters.
    """
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw2, db2, dw1, db1 = backward_prop(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = update_params(dw2, db2, dw1, db1, w1, b1, w2, b2, alpha)
        if i%10 == 0:
            print("iteration: ", i)
            print("accuracy: ", get_accuracy(get_predictions(a2), y))
    return w1, b1, w2, b2

################################
### TESTING HELPER FUNCTIONS ###
################################

def make_predictions(x, w1, b1, w2, b2):
    """
    Helper function to get the predictions.
    """
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, x)
    predictions = get_predictions(a2)
    return predictions

def test_prediction(index, w1, b1, w2, b2):
    """
    Helper function to see the predictions and images.
    """
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], w1, b1, w2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

################
### TRAINING ###
################
w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 1000, 0.1)

##################
### SOME TESTS ###
##################
test_prediction(0, w1, b1, w2, b2)
test_prediction(1, w1, b1, w2, b2)
test_prediction(2, w1, b1, w2, b2)
test_prediction(102, w1, b1, w2, b2)
test_prediction(112, w1, b1, w2, b2) # my model got this one incorrect

#trying on overall testing data
test_predictions = make_predictions(X_test, w1, b1, w2, b2)
get_accuracy(test_predictions, Y_test)

"""
FINAL REFLECTION:
Accuracy when run through 75 times: 87%!!!
Not bad, aside from some tricky numbers. I'd probably need a slightly more complex model for those than relu/gradient descent.
Time spent: 2 hours (and 6 mins spent writing docstrings so it's readable). LFG!
"""