from nn.perceptron import Perceptron
import numpy as np

# Testing the perceptron with AND bitwise dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

print("[INFO] training perceptron...")
# We define the perceptron with the number of columns from the feature vector and the learning rate
p = Perceptron(X.shape[1], alpha=0.1)
# We call fit to train the model, with the specified epochs
p.fit(X, y, epochs=20)

print("[INFO] evaluating perceptron..")
for (x, target) in zip(X, y):
    pred = p.predict(x)
    print("[INFO] data={}, ground-truth={} pred={}".format(x, target[0], pred))