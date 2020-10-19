from __future__ import annotations

import numpy as np


s0_default: float = 1
p_default: float = 0.5

batch_size_default: int = 1

alpha_default: float = 0.1
eps_default: float = 1e-8

mu_default = 1e-2

tolerance_default: float = 1e-3
max_iter_default: int = 1000


class BaseDescent:
    """
    A base class and examples for all functions
    """

    def __init__(self):
        self.w = None

    def step(self, X: np.ndarray, y: np.ndarray, iteration: int) -> np.ndarray:
        """
        Descent step
        :param iteration: iteration number
        :param X: objects' features
        :param y: objects' targets
        :return: difference between weights
        """
        return self.update_weights(self.calc_gradient(X, y), iteration)

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Example for update_weights function
        :param iteration: iteration number
        :param gradient: gradient
        :return: weight difference: np.ndarray
        """
        pass

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Example for calc_gradient function
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        pass


class GradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def __init__(
        self,
        w0: np.ndarray,
        lambda_: float,
        s0: float = s0_default,
        p: float = p_default,
    ):
        """
        :param w0: weight initialization
        :param lambda_: learning rate parameter (float)
        :param s0: learning rate parameter (float)
        :param p: learning rate parameter (float)
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (s0 / (s0 + k)) ** p
        self.w = np.copy(w0)

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Changing weights with respect to gradient
        :param iteration: iteration number
        :param gradient: gradient
        :return: weight difference: np.ndarray
        """
        self.w = self.w - self.eta(iteration) * gradient
        return self.eta(iteration) * gradient

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Getting objects, calculating gradient at point w
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        return 2 * (np.dot(X.T, np.dot(X, self.w) - y)) / len(X)


class StochasticDescent(BaseDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(
        self,
        w0: np.ndarray,
        lambda_: float,
        s0: float = s0_default,
        p: float = p_default,
        batch_size: int = batch_size_default,
    ):
        """
        :param w0: weight initialization
        :param lambda_: learning rate parameter (float)
        :param s0: learning rate parameter (float)
        :param p: learning rate parameter (float)
        :param batch_size: batch size (int)
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (s0 / (s0 + k)) ** p
        self.batch_size = batch_size
        self.w = np.copy(w0)

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Changing weights with respect to gradient
        :param iteration: iteration number
        :param gradient: gradient estimate
        :return: weight difference: np.ndarray
        """
        self.w = self.w - self.eta(iteration) * gradient
        return self.eta(iteration) * gradient

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Getting objects, calculating gradient at point w
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        sample = np.random.randint(X.shape[0], size=self.batch_size)
        return (
            2
            * (np.dot(X[sample].T, np.dot(X[sample], self.w) - y[sample]))
            / self.batch_size
        )


class MomentumDescent(BaseDescent):
    """
    Momentum gradient descent class
    """

    def __init__(
        self,
        w0: np.ndarray,
        lambda_: float,
        alpha: float = alpha_default,
        s0: float = s0_default,
        p: float = p_default,
    ):
        """
        :param w0: weight initialization
        :param lambda_: learning rate parameter (float)
        :param alpha: momentum coefficient
        :param s0: learning rate parameter (float)
        :param p: learning rate parameter (float)
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (s0 / (s0 + k)) ** p
        self.alpha = alpha
        self.w = np.copy(w0)
        self.h = 0

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Changing weights with respect to gradient
        :param iteration: iteration number
        :param gradient: gradient estimate
        :return: weight difference: np.ndarray
        """
        self.h = self.h * self.alpha + self.eta(iteration) * gradient
        self.w -= self.h
        return self.h

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Getting objects, calculating gradient at point w
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        return 2 * np.dot(X.T, (X.dot(self.w) - y)) / len(X)


class Adagrad(BaseDescent):
    """
    Adaptive gradient algorithm class
    """

    def __init__(
        self,
        w0: np.ndarray,
        lambda_: float,
        eps: float = eps_default,
        s0: float = s0_default,
        p: float = p_default,
    ):
        """
        :param w0: weight initialization
        :param lambda_: learning rate parameter (float)
        :param eps: smoothing term (float)
        :param s0: learning rate parameter (float)
        :param p: learning rate parameter (float)
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (s0 / (s0 + k)) ** p
        self.eps = eps
        self.w = np.copy(w0)
        self.g = 0

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Changing weights with respect to gradient
        :param iteration: iteration number
        :param gradient: gradient estimate
        :return: weight difference: np.ndarray
        """
        self.g += np.square(gradient)
        diff = (self.eta(iteration) / np.sqrt(self.eps + self.g)) * gradient
        self.w -= diff
        return diff

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Getting objects, calculating gradient at point w
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        return 2 * np.dot(X.T, (X.dot(self.w) - y)) / len(X)


class GradientDescentReg(GradientDescent):
    """
    Full gradient descent with regularization class
    """

    def __init__(
        self,
        w0: np.ndarray,
        lambda_: float,
        mu: float = mu_default,
        s0: float = s0_default,
        p: float = p_default,
    ):
        """
        :param mu: l2 coefficient
        """
        super().__init__(w0=w0, lambda_=lambda_, s0=s0, p=p)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        l2 = np.sqrt(np.sum(np.power((self.w), 2)))
        return super().calc_gradient(X, y) + l2 * self.mu / 2


class StochasticDescentReg(StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """

    def __init__(
        self,
        w0: np.ndarray,
        lambda_: float,
        mu: float = mu_default,
        s0: float = s0_default,
        p: float = p_default,
        batch_size: int = batch_size_default,
    ):
        """
        :param mu: l2 coefficient
        """
        super().__init__(w0=w0, lambda_=lambda_, s0=s0, p=p, batch_size=batch_size)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        l2 = np.sqrt(np.sum(np.power((self.w), 2)))
        return super().calc_gradient(X, y) + l2 * self.mu / 2


class MomentumDescentReg(MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """

    def __init__(
        self,
        w0: np.ndarray,
        lambda_: float,
        alpha: float = alpha_default,
        mu: float = mu_default,
        s0: float = s0_default,
        p: float = p_default,
    ):
        """
        :param mu: l2 coefficient
        """
        super().__init__(w0=w0, lambda_=lambda_, alpha=alpha, s0=s0, p=p)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        l2 = np.sqrt(np.sum(np.power((self.w), 2)))
        return super().calc_gradient(X, y) + l2 * self.mu / 2


class AdagradReg(Adagrad):
    """
    Adaptive gradient algorithm with regularization class
    """

    def __init__(
        self,
        w0: np.ndarray,
        lambda_: float,
        eps: float = eps_default,
        mu: float = mu_default,
        s0: float = s0_default,
        p: float = p_default,
    ):
        """
        :param mu: l2 coefficient
        """
        super().__init__(w0=w0, lambda_=lambda_, eps=eps, s0=s0, p=p)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        l2 = np.sqrt(np.sum(np.power((self.w), 2)))
        return super().calc_gradient(X, y) + l2 * self.mu / 2


class LinearRegression:
    """
    Linear regression class
    """

    def __init__(
        self,
        descent,
        tolerance: float = tolerance_default,
        max_iter: int = max_iter_default,
    ):
        """
        :param descent: Descent class
        :param tolerance: float stopping criterion for square of euclidean norm of weight difference
        :param max_iter: int stopping criterion for iterations
        """
        self.descent = descent
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Getting objects, fitting descent weights
        :param X: objects' features
        :param y: objects' target
        :return: self
        """
        for iteration in range(0, self.max_iter):
            self.calc_loss(X, y)
            if np.sum(np.power(self.descent.step(X, y, iteration), 2)) < self.tolerance:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Getting objects, predicting targets
        :param X: objects' features
        :return: predicted targets
        """
        return np.dot(X, self.descent.w)

    def calc_loss(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Getting objects, calculating loss
        :param X: objects' features
        :param y: objects' target
        """
        self.loss_history.append(np.mean(np.power(np.dot(X, self.descent.w) - y, 2)))


###########################################################
####################### BONUS TASK ########################
###########################################################


class StochasticAverageGradient(BaseDescent):
    """
    Stochastic average gradient class (BONUS TASK)
    """

    def __init__(
        self,
        w0: np.ndarray,
        lambda_: float,
        x_shape: int,
        s0: float = s0_default,
        p: float = p_default,
    ):
        """
        :param w0: weight initialization
        :param lambda_: learning rate parameter (float)
        :param s0: learning rate parameter (float)
        :param p: learning rate parameter (float)
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (s0 / (s0 + k)) ** p
        self.w = np.copy(w0)
        self.v = np.zeros((x_shape, w0.shape[0]))
        self.d = 0
        self.flag = 0

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Changing weights with respect to gradient
        :param iteration: iteration number
        :param gradient: gradient
        :return: weight difference: np.ndarray
        """
        self.w = self.w - self.eta(iteration) * gradient
        return self.eta(iteration) * gradient

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Getting objects, calculating gradient at point w
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        if self.flag == 0:
            self.flag = 1
            self.d = 2 * (np.dot(X.T, np.dot(X, self.w) - y))
            return self.d / len(X)
        else:
            sample = np.random.randint(X.shape[0], size=1)
            self.d = (
                self.d
                - self.v[sample[0]]
                + 2 * (np.dot(X[sample].T, np.dot(X[sample], self.w) - y[sample]))
            )
            self.v[sample[0]] = 2 * (
                np.dot(X[sample].T, np.dot(X[sample], self.w) - y[sample])
            )
            return self.d / len(X)


###########################################################
####################### BONUS TASK ########################
###########################################################

