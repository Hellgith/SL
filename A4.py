import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

weights = np.zeros(2)
bias = 0
learning_rate = 0.1
epochs = 10

def step_function(x):
    return 1 if x >= 0 else 0

for epoch in range(epochs):
    for i in range(len(X)):
        net_input = np.dot(X[i], weights) + bias
        y_pred = step_function(net_input)

        error = y[i] - y_pred

        weights += learning_rate * error * X[i]
        bias += learning_rate * error

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

x_vals = np.linspace(-0.5, 1.5, 100)

if weights[1] != 0:
    y_vals = -(weights[0] * x_vals + bias) / weights[1]
    plt.plot(x_vals, y_vals)
else:
    plt.axvline(x=-bias/weights[0])

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron Decision Boundary (AND Gate)")
plt.show()

print("Weights:", weights)
print("Bias:", bias)