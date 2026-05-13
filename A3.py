import numpy as np

digits = {
    0: [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    1: [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
    2: [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
    3: [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
    4: [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    5: [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    6: [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    7: [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    8: [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    9: [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
}

labels = {0: -1, 1: 1, 2: -1, 3: 1, 4: -1, 5: 1, 6: -1, 7: 1, 8: -1, 9: 1}

X = np.array([digits[i] for i in range(10)])
y = np.array([labels[i] for i in range(10)])

weights = np.zeros(len(X[0]))
bias = 0
learning_rate = 0.1
epochs = 10

for epoch in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(weights, X[i]) + bias
        predicted = 1 if linear_output >= 0 else -1

        if predicted != y[i]:
            weights += learning_rate * y[i] * np.array(X[i])
            bias += learning_rate * y[i]

def predict_digit(digit):
    input_vector = np.array(digits[digit])
    result = np.dot(weights, input_vector) + bias
    return "Odd" if result >= 0 else "Even"

print("Testing Perceptron Model:")
for digit in range(10):
    print(f"Digit {digit} is {predict_digit(digit)}")