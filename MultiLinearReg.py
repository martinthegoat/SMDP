# coding:utf-8
import numpy as np
from ModelReg import *


class MultiLinear(ModelReg):
    def __init__(self, name):
        super(MultiLinear, self).__init__(name)
        self.model_name = name
        self.theta = None

    def compute_cost(self, X, y, theta):
        """
        Compute Mean Squared Error (MSE).

        Input Parameters
        ----------------
        X: 2D feature ndarray. Dimension(m x n)
        y: 1D array. dimension(1 x m)
        theta : 1D array of fitting parameters or weights. Dimension (1 x n)

        Output Parameters
        -----------------
        result : Scalar value.
        """
        predictions = X @ theta  # @ = element wise dot production, same as 'X.dot(theta)'
        errors = predictions - y
        sqrErrors = errors ** 2
        result = sum(sqrErrors) / len(y)
        return result

    def gradient_descent(self, X, y, theta, alpha, iterations):
        """
        Apply derivative function on cost function, so that errors reduce.

        Input Parameters
        ----------------
        X: 2D feature ndarray. Dimension(m x n)
        y: 1D array. dimension(m x 1)
        theta : 1D array of fitting parameters or weights. Dimension (1 x n)
        alpha : Learning rate. Scalar value
        iterations: Number of iterations. Scalar value.

        Output Parameters
        -----------------
        theta : Final Value. 1D array of fitting parameters or weights. Dimension (1 x n)
        cost_history: Conatins value of cost for each iteration. 1D array. Dimansion(m x 1)
        """
        cost_history = np.zeros(iterations)

        for i in range(iterations):
            predictions = X @ theta
            errors = predictions - y
            sum_delta = (alpha / len(y)) * (X.T @ errors)
            theta = theta - sum_delta

            cost_history[i] = self.compute_cost(X, y, theta)

        return theta, cost_history

    def fit(self, X_train, y_train, iterations=400, alpha=0.15):
        """
        Fit the data for training.

        Input Parameters
        ----------------
        X_train: 2D feature ndarray. Dimension(m x n)
        y_train: 1D array. dimension(m x 1)
        iterations: Number of iterations. Scalar value.
        alpha : Learning rate. Scalar value

        Output Parameters
        -----------------
        theta : 1D array of fitting parameters or weights. Dimension (1 x n)
        cost_history: Conatins value of cost for each iteration. 1D array. Dimansion(m x 1)
        """
        # Number of training examples
        m = len(y_train)

        # use hstack() function from numpy to add column of ones to X feature
        X_train = np.hstack((np.ones((m, 1)), X_train))

        # calculate theta parameters
        theta = np.zeros(X_train.shape[1])

        theta, cost_history = self.gradient_descent(X_train, y_train, theta, alpha, iterations)
        self.theta = theta
        return theta, cost_history

    def predict(self, X_test, theta=None):
        """
        Fit the data for training.

        Input Parameters
        ----------------
        X_test: 2D feature ndarray. Dimension(m x n)
        theta : 1D array of fitting parameters or weights. Dimension (1 x n)

        Output Parameters
        -----------------
        model_pred: list of predicted values
        """
        model_pred = []
        if theta is None:
            theta = self.theta
        for index in range(len(X_test)):
            normalize_test_data = np.hstack((np.ones(1), X_test.iloc[index]))
            pred = normalize_test_data.dot(theta)
            model_pred.append(pred)
        return np.array(model_pred)
