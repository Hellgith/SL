import numpy as np

patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
])

n = patterns.shape[1]
W = np.zeros((n, n))

for p in patterns:
    W += np.outer(p, p)

np.fill_diagonal(W, 0)

print("Weight Matrix:\n", W)

def recall(x, steps=5):
    x = x.copy()
    for _ in range(steps):
        x = np.sign(np.dot(W, x))
        x[x == 0] = 1
    return x

test = np.array([1, -1, 1, 1])
result = recall(test)

print("\nTest Input:", test)
print("Recalled Pattern:", result)