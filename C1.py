import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = Net()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(2000):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Final Output:\n", model(X).detach())


import tensorflow as tf
from sklearn.metrics import accuracy_score

X = tf.constant([[0,0],[0,1],[1,0],[1,1]], dtype=tf.float32)
y = tf.constant([[0],[1],[1],[1]], dtype=tf.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=100, verbose=0)

pred = model.predict(X)
pred_class = (pred > 0.5).astype(int)

print("Predictions:\n", pred_class)
print("Accuracy:", accuracy_score(y, pred_class))