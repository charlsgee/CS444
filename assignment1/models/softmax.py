"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        N, D = X_train.shape
        scores = X_train @ self.w 
        
        expscores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  
        probs = expscores / np.sum(expscores, axis=1, keepdims=True)  

        labels = np.zeros_like(probs)
        labels[np.arange(N), y_train] = 1
        grad = (X_train.T @ (probs - labels)) / N  
        grad += self.reg_const * self.w 

        return grad
        

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.
        
        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape
        self.w = 0.01 * np.random.uniform(-1, 1, (D, self.n_class))
        for epoch in range(self.epochs):
            idx = np.random.permutation(N)
            xshuffle = X_train[idx]
            yshuffle = y_train[idx]

            for i in range(0, N, 128):
                xbatch1 = xshuffle[i:i + 128]
                ybatch = yshuffle[i:i + 128]

                grad = self.calc_gradient(xbatch1, ybatch)
                self.w -= self.lr * grad 
                
            self.lr *= .95

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        scores = X_test @ self.w  
        return np.argmax(scores, axis=1)
