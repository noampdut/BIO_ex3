import numpy as np
import sys

weights_file = sys.argv[1]
test_file = sys.argv[2]


'''
This function calculate sigmoid activation function
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


lines = None
# Load the saved weights from the file
with open(weights_file, "r") as file:
    lines = file.readlines()

w1 = np.loadtxt(lines[1:17])
b1 = np.loadtxt(lines[19:27])
w2 = np.loadtxt(lines[29:37])


data = []
# Load data
with open(test_file, "r") as file:
    for line in file:
        bit_string = line.strip()
        bit_string = [int(bit) for bit in bit_string]
        data.append(bit_string)


# Write predictions to file
with open("predictions0.txt", "w") as file:
    # Make Prediction
    for x in data:
        hidden = sigmoid(np.dot(x, w1) + b1)
        output = sigmoid(np.dot(hidden, w2))
        prediction = int(output > 0.5)

        file.write(str(prediction) + "\n")
