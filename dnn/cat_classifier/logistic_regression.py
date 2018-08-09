import h5py
import numpy as np
import os
from PIL import Image

def load_dataset(folder):
    '''
    This function loads the h5 dataset of cat versus non-cat images

    Arguments:
    folder -- location of the test and training sets

    Returns:
    train_set_x_orig -- input features of examples from the train set
    train_set_y_orig -- labels of examples from the training set
    test_set_x_orig -- input features of examples from the test set
    test_set_y_orig -- labels of examples from the test set
    classes -- number of label categories
    '''
    with h5py.File(os.path.join(folder, 'train_catvnoncat.h5'), 'r') as train_dataset:
        train_set_x_orig = np.array(train_dataset['train_set_x'][:])
        train_set_y_orig = np.array(train_dataset['train_set_y'][:])

    with h5py.File(os.path.join(folder, 'test_catvnoncat.h5'), 'r') as test_dataset:
        test_set_x_orig = np.array(test_dataset['test_set_x'][:])
        test_set_y_orig = np.array(test_dataset['test_set_y'][:])
        classes = np.array(test_dataset['list_classes'][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def initialize_with_zeros(dim):
    '''
    This function creates a vector of zeros of shape (dim, 1) for weights w
    and initializes bias b to 0.

    Arguments:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    '''
    w = np.zeros((dim, 1))
    b = 0
    return w, b

def sigmoid(z):
    '''
    This function computes the sigmoid of z.

    Arguments:
    z -- A scalar or numpy n-dimensional array of any size

    Returns:
    s -- sigmoid(z)
    '''
    s = 1. / (1 + np.exp(-z))
    return s

def propogate(w, b, X, Y):
    '''
    This function calculates prediction (x=>z=>a) and cost L(a, y) during
    forward prop and gradients in a single iteration over all examples

    Arguments:
    w -- weights, a numpy array of shape (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples). In other words,
    input features of all examples are stacked horizontally
    Y -- ground truth label vector {0 (non-cat), 1 (cat)} of shape (1, number of examples)

    Returns:
    cost -- negative log-likelihood cost for logistric regression. To explain,
    if y = 0, according to the loss formula, the cost is -log(1-a). Therefore,
    the cost is minimized (approaches 0) when 'a' approaches 0 which is what y is.
    Similarily, if y = 1, the cost is negative log(a). Therefore, the cost is
    minimized (approaches 0) when a approaches 1 which is what y is.
    grads -- gradients of the cost function with respect to w and b parameters
    '''
    m = X.shape[1]

    # forward prop
    A = sigmoid(np.dot(w.T, X) + b)
    cost = - np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) / m
    # removes single-dimensional entries from the shape of an array.
    # i.e., array of shape (1, 3, 1) becomes an array of shape (3,)
    cost = np.squeeze(cost)
    
    # backward prop
    dw = np.dot(X, (A-Y).T) / m     # how gradient (slope) of the cost function reacts to every w
    db = np.sum(A-Y) / m            # gradient (slope) of cost fuction with respect to b

    grads = { 'dw': dw, 'db': db }
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=True):
    '''
    This function optimizes model parameters w and b iteratively (num_iterations)

    Arguments:
    w -- weights, a numpy array of shape (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples). In other words,
    input features of all examples are stacked horizontally
    Y -- ground truth label vector {0, 1} of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule. How big or small
    of a step to take during each iteration
    print_cost -- True to print the loss every 100 steps

    Returns:
    The following values indicate the state of the model after num_iterations
    params -- learned weights w and bias b (python dict) after the optimization loop
    grads -- last known gradients (dw and db) of the cost function with respect to w and b
    cost -- list of all the costs computed during the optimization, this will be used to plot the learning curve
    '''
    costs = []
    for i in range(num_iterations):
        grads, cost = propogate(w, b, X, Y)    # it does both forward and backward propogations
        dw = grads['dw']
        db = grads['db']
        w -= learning_rate * dw        # updates the weights
        b -= learning_rate * db        # updates the bias term

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    params = { 'w': w, 'b': b }
    grads = { 'dw': dw, 'db': db }
    return params, grads, costs

def predict(w, b, X):
    '''
    This function predicts (calculates probabilities) whether x is a cat using the
    weights w and bias b.

    Arguments:
    w -- learned weights, a numpy array of shape (num_px * num_px * 3, 1)
    b -- learned bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples). In other words,
    input features of all examples are stacked horizontally

    Returns:
    Y_prediction -- numpy array (vector) of shape (1, number of examples)
    containing all predictions {0 for non-cat and 1 for cat} for the examples in X
    '''
    m = X.shape[1]   # number of examples to predict
    Y_prediction = np.zeros((1, m))
    A = sigmoid(np.dot(w.T, X) + b)     # returns probabilities for each example
    Y_prediction = np.around(A)        # convert probabilities to either 0 or 1
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=True):
    """
    Builds the logistic regression model by calling the functions implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model
    """

    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])
    
    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve learned parameters w and b
    w = parameters['w']
    b = parameters['b']

    # Predict test/train set examples
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # Print train/test errors
    print("train accuracy: {}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = { "costs": costs,
    "Y_prediction_test": Y_prediction_test,
    "Y_prediction_train" : Y_prediction_train,
    "w" : w,
    "b" : b,
    "learning_rate" : learning_rate,
    "num_iterations": num_iterations }
    
    return d

def main():
    # load dataset
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset('datasets')
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    print ("Number of training examples: m_train = %i " % m_train)
    print ("Number of testing examples: m_test = %i" % m_test)
    print ("Height/Width of each image: num_px = %i" % num_px)
    print ("Each image is of size: (%i, %i, 3)" % (num_px, num_px))
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))

    # reshape data (image to vector)
    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

    # normailze data
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    # build model
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1000, learning_rate=0.005, print_cost=True)

    # change 'i' to show some example predictions
    i = 20
    Image.fromarray(train_set_x_orig[i], 'RGB').show()

    x = train_set_x[:, i, None]
    if predict(d['w'], d['b'], x):
        print('Image %i is a cat' % (i+1))
    else:
        print('Image %i is NOT a cat' % (i+1))

main()