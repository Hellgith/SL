import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 400)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sigmoid_output = sigmoid(x)

plt.figure(figsize=(8, 6))
plt.plot(x, sigmoid_output, label='Sigmoid', color='blue')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.grid(True)
plt.axhline(0.5, color='gray', linestyle='--')
plt.axvline(0, color='black', linestyle='--')
plt.legend()
plt.show()


def relu(x):
    return np.maximum(0, x)

relu_output = relu(x)

plt.figure(figsize=(8, 6))
plt.plot(x, relu_output, label='ReLU', color='green')
plt.title('ReLU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.grid(True)
plt.axvline(0, color='black', linestyle='--')
plt.legend()
plt.show()


def tanh(x):
    return np.tanh(x)

tanh_output = tanh(x)

plt.figure(figsize=(8, 6))
plt.plot(x, tanh_output, label='Tanh', color='purple')
plt.title('Tanh Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.grid(True)
plt.axhline(0, color='black', linestyle='--')
plt.axhline(1, color='gray', linestyle='--')
plt.axhline(-1, color='gray', linestyle='--')
plt.legend()
plt.show()


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

leaky_relu_output = leaky_relu(x)

plt.figure(figsize=(8, 6))
plt.plot(x, leaky_relu_output, label='Leaky ReLU', color='red')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.grid(True)
plt.axvline(0, color='black', linestyle='--')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 400)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sigmoid_output = sigmoid(x)

plt.figure(figsize=(8, 6))
plt.plot(x, sigmoid_output, label='Sigmoid', color='blue')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.grid(True)
plt.axhline(0.5, color='gray', linestyle='--')
plt.axvline(0, color='black', linestyle='--')
plt.legend()
plt.show()


def relu(x):
    return np.maximum(0, x)

relu_output = relu(x)

plt.figure(figsize=(8, 6))
plt.plot(x, relu_output, label='ReLU', color='green')
plt.title('ReLU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.grid(True)
plt.axvline(0, color='black', linestyle='--')
plt.legend()
plt.show()


def tanh(x):
    return np.tanh(x)

tanh_output = tanh(x)

plt.figure(figsize=(8, 6))
plt.plot(x, tanh_output, label='Tanh', color='purple')
plt.title('Tanh Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.grid(True)
plt.axhline(0, color='black', linestyle='--')
plt.axhline(1, color='gray', linestyle='--')
plt.axhline(-1, color='gray', linestyle='--')
plt.legend()
plt.show()


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

leaky_relu_output = leaky_relu(x)

plt.figure(figsize=(8, 6))
plt.plot(x, leaky_relu_output, label='Leaky ReLU', color='red')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.grid(True)
plt.axvline(0, color='black', linestyle='--')
plt.legend()
plt.show()