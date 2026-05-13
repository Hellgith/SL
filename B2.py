import numpy as np

class ART1:
    def __init__(self, n_features, n_clusters, vigilance=0.5):
        self.n = n_features
        self.m = n_clusters
        self.vigilance = vigilance

        self.bottom_up = np.ones((self.m, self.n)) / (1 + self.n)
        self.top_down = np.ones((self.m, self.n))

    def train(self, X):
        for i, x in enumerate(X):
            print(f"\nInput {i+1}: {x}")

            choice = np.dot(self.bottom_up, x)

            while True:
                j = np.argmax(choice)

                match = np.sum(np.minimum(x, self.top_down[j])) / np.sum(x)

                if match >= self.vigilance:
                    print(f"Assigned to cluster {j}")

                    self.top_down[j] = np.minimum(x, self.top_down[j])
                    self.bottom_up[j] = self.top_down[j] / (0.5 + np.sum(self.top_down[j]))
                    break
                else:
                    choice[j] = -1

X = np.array([
    [1, 0, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 1]
])

art = ART1(n_features=4, n_clusters=3, vigilance=0.6)
art.train(X)