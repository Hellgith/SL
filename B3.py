import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)

w1 = np.random.rand(2, 2)
w2 = np.random.rand(2, 1)
b1 = np.random.rand(1, 2)
b2 = np.random.rand(1, 1)

lr = 0.1

for epoch in range(5000):
    h = sigmoid(np.dot(X, w1) + b1)
    o = sigmoid(np.dot(h, w2) + b2)

    error = y - o
    d_o = error * d_sigmoid(o)
    d_h = d_o.dot(w2.T) * d_sigmoid(h)

    w2 += h.T.dot(d_o) * lr
    w1 += X.T.dot(d_h) * lr
    b2 += np.sum(d_o, axis=0, keepdims=True) * lr
    b1 += np.sum(d_h, axis=0, keepdims=True) * lr

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

print("\nFinal Output:")
print(o)