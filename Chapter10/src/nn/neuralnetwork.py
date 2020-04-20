import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        """
        The constructor of the Neural Network
        -----------------------------------
        Each neural network consists of input nodes, at least 1 hidden layer and an output layer

        :param layers: list of integers representing the architecture of the NN, e.g. [2, 2, 1] - 2 input nodes 1 hidden
        layer with 2 nodes and 1 ouput layer with 1 node
        :param alpha: learning rate
        """
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # Weight matrix initialization for each of the layers
        for i in np.arange(0, len(layers) - 2):
            # we initialized the weight matrix with random sample values of the normal distribution
            # the Weight matrix will be MxN such that we can connect each of the nodes of the current layer
            # to each of the nodes to the next layer
            # If layers[i] = 2 and layers[i+1] = 2 -> W will be a matrix 2x2.
            # We also add one to the number of the current layer (layer[i]) and 1 to the next (layer[i+1]) to
            # account for the bias, ultimately W = 3x3 matrix
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            # weight matrix normalization
            self.W.append(w / np.sqrt(layers[i]))

        # This accounts to the special case of the last two layers in the network
        # layers[-2] - second to last layer needs only bias in the input but not in the output
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "Neural network: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        """
        Sigmoid activation function
        :param x: input weighted vector
        :return: activated value
        """
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        """
        Used in the backpropagation phase of the neural networking using the backpropagating algorithm
        :param x:
        :return:
        """
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        """
        Trains the model with the specified dataset and corresponding target labels
        :param X: dataset
        :param y: target labels
        :param epochs: # of epoch for training
        :param displayUpdate: parameter to adjust the update information on console
        """
        # Adding extra column to each of the datapoints for the bias trick
        # to be treated as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over each epoch
        for epoch in np.arange(0, epochs):
            for(x, target) in zip(X, y):
                # loop over each data point and train the network
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format((epoch + 1), loss))

    def fit_partial(self, x, y):
        """
        Partially fitting our model
        :param x: data point from the dataset
        :param y: corresponding target label
        """
        # List of outputs of the activiations from each layer
        # the first activiation is the input itself (data point)
        A = [np.atleast_2d(x)]

        # Feedforward pass
        # we pass the data point thrugh each of the layers in the network
        # and each activation is then passed to the next layer in the network, dotted with the corresponding
        # weight matrix of the layer
        # Each activation output is stored in the A list
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)

            A.append(out)

        # Backpropagation phase
        # --------------------
        # Compute the differnce between the prediction (the final output in the list of activations (A))
        # and the actual target label
        error = A[-1] - y

        # We start with initializing a list D, which contains the deltas for the chain rule
        # The first element is the error times the derivative of the output of the last layer
        D = [error * self.sigmoid_deriv(A[-1])]
        # We then start iterating each layer backwards aplying the chain rule
        # We ignore the last two layers since they are already taken care of (the first elemnt in D list)
        for layer in np.arange(len(A) - 2, 0, -1):
            # The delta for the current layer is computed by dotting the delta of the previous layer with
            # the weight matrix of the current layer, which is then multiplied with the derivative of the
            # activation function for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # reverse the deltas, becuase of the reversed loop
        D = D[::-1]

        # update the weight matrices for each layer
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        """
        Predicts the class label for the given test vector
        :param X: the test vector
        :param addBias: wether to add extra column for the bias
        :return: prediction
        """
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        """
        Calculates the loss of the whole dataset, used in each epoch to visualize the improvement over time
        :param X: dataset
        :param targets: class labels
        :return: loss
        """
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss
