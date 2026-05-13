import numpy as np

def andnot_mcp(x1, x2):
    w1 = 1
    w2 = -1
    threshold = 1

    net = x1 * w1 + x2 * w2

    if net >= threshold:
        return 1
    else:
        return 0

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

print("Truth Table of ANDNOT using McCulloch-Pitts Neural Network")
print("x1 x2 Y")
print("----------------")

for x in inputs:
    x1 = x[0]
    x2 = x[1]
    y = andnot_mcp(x1, x2)
    print(x1, " ", x2, " ", y)