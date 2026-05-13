import numpy as np

class BAM:
    def __init__(self, input_size, output_size):
        self.weights = np.zeros((input_size, output_size))

    def train(self, input_patterns, output_patterns):
        for x, y in zip(input_patterns, output_patterns):
            self.weights += np.outer(x, y)
        print("Updated Weight Matrix:\n", self.weights)

    def recall(self, input_pattern, direction="forward"):
        if direction == "forward":
            raw_output = np.dot(input_pattern, self.weights)
        else:
            raw_output = np.dot(input_pattern, self.weights.T)

        output = np.sign(raw_output)
        output[output == 0] = 1
        return output

input_patterns = np.array([
    [1, -1, 1],
    [-1, 1, -1]
])

output_patterns = np.array([
    [1, -1],
    [-1, 1]
])

bam = BAM(input_size=3, output_size=2)
bam.train(input_patterns, output_patterns)

test_input = np.array([1, -1, 1])
retrieved_output = bam.recall(test_input, direction="forward")
print("\nRecalled Output for input {}: {}".format(test_input, retrieved_output))

test_output = np.array([1, -1])
retrieved_input = bam.recall(test_output, direction="backward")
print("\nRecalled Input for output {}: {}".format(test_output, retrieved_input))