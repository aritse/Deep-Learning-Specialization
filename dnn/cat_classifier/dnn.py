import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def load_dataset(catnotcat):
    with h5py.File(os.path.join(catnotcat, 'train_catvnoncat.h5')) as train_dataset:
        train_x = np.array(train_dataset['train_set_x'][:])
        train_y = np.array(train_dataset['train_set_y'][:])
        train_y = train_y.reshape((1, train_y.shape[0]))
    with h5py.File(os.path.join(catnotcat, 'test_catvnoncat.h5')) as test_dataset:
        test_x = np.array(test_dataset['test_set_x'][:])
        test_y = np.array(test_dataset['test_set_y'][:])
        test_y = test_y.reshape((1, test_y.shape[0]))
    return train_x, train_y, test_x, test_y

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * (s * (1 - s))
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.copy(dA)
    dZ[Z <= 0] = 0
    return dZ

def get_shape(d):
    if isinstance(d, dict):
        return { k: get_shape(d[k]) for k in d }
    return None

def shape_equal(a, b):
    return get_shape(a) == get_shape(b)

def plot_costs(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Steps')
    plt.title('Learning rate = '+str(learning_rate))
    plt.show()

def peek(var):
    print(type(var))
    if isinstance(var, np.ndarray):
        print(var.shape)
    print(var)
    exit()

def parameters_to_vector(parameters):
    theta = np.ndarray((0, 1))
    for val in parameters.values():
        vector = np.reshape(val, (-1, 1))
        theta = np.concatenate((theta, vector))
    return theta

def gradients_to_vector(gradients):
    d_theta = np.ndarray((0, 1))
    for key, val in gradients.items():
        if 'dW' in key or 'db' in key:
            vector = np.reshape(val, (-1,1))
            d_theta = np.concatenate((d_theta, vector))
    return d_theta

def vector_to_theta_parameters(vals, parameters):
    theta_parameters = {}
    start = 0
    end = 0
    for key, val in parameters.items():
        size = val.shape[0] * val.shape[1]
        end = start + size
        theta_parameters[key] = np.ndarray(val.shape, buffer=vals[start:end])
        start = end
    return theta_parameters

def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.3
        parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def gradient_check(parameters, gradients, X, Y, epsilon, l2_parameter):
    parameter_values = parameters_to_vector(parameters)
    gradient_values = gradients_to_vector(gradients)
    assert(parameter_values.shape == gradient_values.shape)
    num_parameters = parameter_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    print('Number of parameters: ' + str(num_parameters))
    for i in range(num_parameters):
        if i % 1000 == 0:
            progress = i * 100 // num_parameters
            print('Progress: ' + str(progress))
        theta_plus = np.copy(parameter_values)
        theta_plus[i] = theta_plus[i] + epsilon
        theta_plus_parameters = vector_to_theta_parameters(theta_plus, parameters)
        assert(shape_equal(parameters, theta_plus_parameters))
        AL, _ = model_forward(X, theta_plus_parameters)
        J_plus[i] = compute_cost(AL, Y, theta_plus_parameters, l2_parameter)

        theta_minus = np.copy(parameter_values)
        theta_minus[i] = theta_minus[i] - epsilon
        theta_minus_parameters = vector_to_theta_parameters(theta_minus, parameters)
        assert(shape_equal(parameters, theta_minus_parameters))
        AL, _ = model_forward(X, theta_minus_parameters)
        J_minus[i] = compute_cost(AL, Y, theta_minus_parameters, l2_parameter)

        gradapprox[i] = 0.5 * (J_plus[i] - J_minus[i]) / epsilon

    numerator = np.linalg.norm(gradient_values - gradapprox)
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(gradient_values)
    diff = numerator / denominator
    return diff

def predict(X, parameters):
    AL, _ = model_forward(X, parameters)
    return AL

def update_parameters(parameters, gradients, alpha):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters['W'+str(l)] = parameters['W'+str(l)] - alpha * gradients['dW'+str(l)]
        parameters['b'+str(l)] = parameters['b'+str(l)] - alpha * gradients['db'+str(l)]
    return parameters

def linear_backward(dZ, cache):
    A_prev, W, _ = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def regularization(parameters, l2_parameter, m):
    penalty = 0
    for key, val in parameters.items():
        if key.startswith('W'):
            penalty += np.sum(np.square(val))
    penalty *= (l2_parameter / (2 * m))
    return penalty

def compute_cost(AL, Y, parameters, l2_parameter):
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cross_entropy_cost = np.sum(logprobs) / m
    penalty = regularization(parameters, l2_parameter, m)
    cost = cross_entropy_cost + penalty
    return cost

def model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)
    return AL, caches

def model(X, Y, layer_dims, alpha=1e-2, steps=2000, print_cost=True, l2_parameter=0):
    costs = []
    parameters = initialize_parameters(layer_dims)
    for i in range(steps):
        AL, caches = model_forward(X, parameters)
        cost = compute_cost(AL, Y, parameters, l2_parameter)
        gradients = model_backward(AL, Y, caches, l2_parameter)
        # if i == 250:
        #     epsilon = 1e-7
        #     print('Checking gradients ...')
        #     diff = gradient_check(parameters, gradients, X, Y, epsilon, l2_parameter)
        #     if diff > (2 * epsilon):
        #         print('Mistake in the backward propagation: difference = '+str(diff))
        #     else:
        #         print('No mistake in the backward propagation: difference = '+str(diff))
        if print_cost and i % 50 == 0:
            print('Cost after step {}: {:.3f}'.format(i, cost))
            costs.append(cost)
        parameters = update_parameters(parameters, gradients, alpha)
    plot_costs(costs, alpha)
    return parameters
    
def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache
    
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache

def model_backward(AL, Y, caches, l2_parameter):
    m = Y.shape[1]
    gradients = {}
    L = len(caches)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    W = current_cache[0][1]
    dA_prev_temp, dW_temp, db_temp =  linear_activation_backward(dAL, current_cache, 'sigmoid')
    gradients['dA'+str(L-1)] = dA_prev_temp
    gradients['dW'+str(L)] = dW_temp  + ((l2_parameter * W) / m)
    gradients['db'+str(L)] = db_temp
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        W = current_cache[0][1]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(gradients['dA'+str(l+1)], current_cache, 'relu')
        gradients['dA'+str(l)] = dA_prev_temp
        gradients['dW'+str(l+1)] = dW_temp  + ((l2_parameter * W) / m)
        gradients['db'+str(l+1)] = db_temp
    return gradients

def main():
    train_x, train_y, test_x, test_y = load_dataset('catnotcat')
    m_train = train_x.shape[0]
    m_test = test_x.shape[0]

    train_x = train_x.reshape(m_train, -1).T
    test_x = test_x.reshape(m_test, -1).T

    train_x = train_x / 255
    test_x = test_x / 255

    layer_dims = [12288, 10, 5, 1]
    parameters = model(train_x, train_y, layer_dims)

    train_predictions = predict(train_x, parameters)
    train_predictions[train_predictions > 0.5] = 1
    train_predictions[train_predictions <= 0.5] = 0
    accuracy = np.sum(train_predictions == train_y) / m_train
    print('Train accuracy: ' + str(accuracy))

    test_predictions = predict(test_x, parameters)
    test_predictions[test_predictions > 0.5] = 1
    test_predictions[test_predictions <= 0.5] = 0
    accuracy = np.sum(test_predictions == test_y) / m_test
    print('test accuracy: ' + str(accuracy))

main()