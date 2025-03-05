"""Support Vector Machine (SVM) model."""

import numpy as np
import matplotlib.pyplot as plt


class SVM:
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
        self.loss_history = []  # To store loss values for each epoch
        self.accuracy_history = []  # To store accuracy values for each epoch
    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        N = X_train.shape[0]
        grad = np.zeros(self.w.shape)
        for i in range(N):
            vals = np.dot(X_train[i], self.w.T)
            truevals = vals[y_train[i]]
            for j in range(self.n_class):
                if j == y_train[i]:
                    continue
                margin = vals[j] - truevals + 1 
                if margin > 0:
                    grad[j, :] += X_train[i]
                    grad[y_train[i], :] -= X_train[i]
        grad /= N
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
        # TODO: implement me
        N, D = X_train.shape
        self.w = 0.01 * np.random.randn(self.n_class, D)

        for epoch in range(self.epochs):
            idx = np.arange(N)
            np.random.shuffle(idx)
            X_train = X_train[idx]
            y_train = y_train[idx]

            for i in range(0, N, 128):
                xbatch1 = X_train[i:i + 128]
                ybatch = y_train[i:i + 128]

                grad = self.calc_gradient(xbatch1, ybatch)
                self.w -= self.lr * grad 
            # Calculate and store loss and accuracy for this epoch
            loss = self.calc_loss(X_train, y_train)
            accuracy = self.calc_accuracy(X_train, y_train)
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            self.lr *= .2
        return

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
        # TODO: implement me
        vals = np.dot(X_test, self.w.T)
        return np.argmax(vals, axis=1)
    def calc_loss(self, X: np.ndarray, y: np.ndarray) -> float:
       
        N = X.shape[0]
        vals = np.dot(X, self.w.T)
        truevals = vals[np.arange(N), y]
        margins = np.maximum(0, vals - truevals[:, np.newaxis] + 1)
        margins[np.arange(N), y] = 0  # Ignore the correct class
        loss = np.sum(margins) / N
        loss += 0.5 * self.reg_const * np.sum(self.w * self.w)  # Regularization term
        return loss

    def calc_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100

    def plot_loss(self):
        plt.figure(figsize=(10, 6))

        plt.plot(range(1, self.epochs + 1), self.loss_history, marker='.', linestyle='-', color='b', label='Loss')
        plt.xticks(np.arange(0, self.epochs + 1, step=self.epochs//10), fontsize=12, rotation=45)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title("SVM Training Loss", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.show()

    def plot_accuracy(self):
        plt.figure(figsize=(12, 6))

        plt.plot(range(1, self.epochs + 1), self.accuracy_history, marker='o', linestyle='-', color='g', label='Accuracy')

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Accuracy (%)", fontsize=14)
        plt.title("SVM Training Accuracy", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.xticks(np.arange(0, self.epochs + 1, step=self.epochs//(self.epochs//10)), fontsize=12)  # Add ticks every 50 epochs
        plt.yticks(fontsize=12)

        plt.legend(fontsize=12)

        plt.show()