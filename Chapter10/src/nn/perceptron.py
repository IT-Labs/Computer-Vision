import numpy as np


class Perceptron:
    def __init__(self, N, alpha=0.1):
        """
        The constructor of the perceptron
        :param N: number of columns in the feature vector
        :param alpha: learning rate (common choices - 0.1, 0.01, 0.001)
        """
        # Initializing the weight matrix with random values from a normal ("Gaussian") distribution
        # the weight matrix will have N + 1 entries, one for each of the columns + 1 for the bias
        # np.sqrt(N) - common technique to scale the weight matrix -> faster convergence
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        """
        The step acitivation function
        :param x: weighted vector
        :return: predictions
        """
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        """
        This function is used to train our model with the supplied dataset and corresponding target labels
        :param X: the training dataset
        :param y: the corresponding target labels for each of the data points in the dataset
        :param epochs: number of epochs the perceptron should be trained for
        """
        # applying the bias trick by adding extra column to each of the data points
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                # applying the step function to the dot product of each of the data point with the weigh matrix
                # that gives a prediction
                p = self.step(np.dot(x, self.W))

                if p != target:
                    # compute the error
                    error = p - target

                    # update the weight matrix
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        """
        Predicts the class labels for a given set of input data
        :param X: the set of data points to be classified
        :param addBias: wether a bias should be added to the datapoints
        :return: predictions for each of the data points
        """
        # we ensure that the input dataset is at least  a 2-Dimensional matrix
        X = np.atleast_2d(X)

        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))
