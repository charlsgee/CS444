"""Perceptron model."""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
def get_acc(pred, y_test):
    return np.sum(y_test == pred) / len(y_test) * 100
class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.loss = []
        self.accuracy = []
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        self.w = .01*np.random.uniform(low=-1.0, high=1.0, size=(self.n_class, X_train.shape[1]))
        for epoch in tqdm(range(self.epochs), desc="Training", unit="epoch"):
            misses = 0
            for x, y in zip(X_train, y_train):
                ynext = np.argmax(self.w @ x)
                if y != ynext:
                    misses += 1
                    self.w[ynext, :] -= self.lr * x
                    self.w[y, :] += self.lr * x
            self.loss.append(misses)
            self.lr *= .2
            y_pred_all = self.predict(X_train)
            accuracy = get_acc(y_pred_all, y_train)
            self.accuracy.append(accuracy)
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
        ypred = np.argmax((X_test @ self.w.T),axis=1)

        return ypred
    
    def plot_loss(self):

        plt.figure(figsize=(10, 6))

        plt.plot(range(1, self.epochs + 1), self.loss, marker='.', linestyle='-', color='b', label='Misclassifications')
        plt.xticks(np.arange(0, self.epochs + 1, step=self.epochs//10), fontsize=12, rotation=45)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Misclassifications", fontsize=14)
        plt.title("Perceptron Training Loss (Misclassifications)", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.show()
    def plot_accuracy(self):
        if len(self.accuracy) != self.epochs:
            raise ValueError(
                f"Length of accuracy_history ({len(self.accuracy)}) does not match "
                f"the number of epochs ({self.epochs}). Check the training loop."
            )

        plt.figure(figsize=(12, 6))

        plt.plot(range(1, self.epochs + 1), self.accuracy, marker='o', linestyle='-', color='g', label='Accuracy')

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Accuracy (%)", fontsize=14)
        plt.title("Perceptron Training Accuracy", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.xticks(np.arange(0, self.epochs + 1, step=self.epochs//(self.epochs//10)), fontsize=12)  # Add ticks every 50 epochs
        plt.yticks(fontsize=12)

        plt.legend(fontsize=12)

        plt.show()