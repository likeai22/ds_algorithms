import numpy as np
import pandas as pd


class MyLinearRegression:
    def __init__(self, samples: pd.DataFrame, targets: pd.DataFrame,
                 fit_intercept: bool = True, copy: bool = True):
        self.weight = None
        self.fit_intercept = fit_intercept
        self.samples = samples.copy() if copy else samples
        self.samples = np.hstack((self.samples, np.ones((self.samples.shape[0], 1))))
        self.targets = targets

    def fit(self):
        self.weight = np.linalg.inv(
            self.samples.T @ self.samples) @ self.samples.T @ self.targets
        return self

    def predict(self):
        return self.samples @ self.weight

    def get_weights(self):
        return self.weight


class MyGradientLinearRegression(MyLinearRegression):
    def __init__(self,
                 iters: int = int(1e6),
                 alpha: float = 1e-3,
                 diff_mse: float = 1e-5,
                 print_cost: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.iters = iters
        self.alpha = alpha
        self.diff_mse = diff_mse
        self.weight = np.ones(self.samples.shape[1])
        self.print_cost = print_cost
        self.loss_dict = {}

    def mean_squared_error(self):
        loss = self.samples @ self.weight - self.targets
        return np.mean(np.square(loss))

    def fit(self):
        previous_cost = self.mean_squared_error()
        current_cost = previous_cost

        self.loss_dict[0] = previous_cost

        w = self.weight

        for count in range(1, self.iters + 1):

            new_w = w - self.alpha * self._calc_gradient()

            if count % 100 == 0 and self.print_cost is True:
                print(f"Cost at iteration {count} is {current_cost}, weight={self.weight}")

            w = new_w
            self.weight = w

            current_cost = self.mean_squared_error()
            self.loss_dict[count] = current_cost

            if np.abs(current_cost - previous_cost) < self.diff_mse:
                break

            previous_cost = current_cost

        print(f'Model alpha: {self.alpha}, diff_mse: {self.diff_mse}, iterations: {count} ...')

    def _calc_gradient(self):
        pred = self.samples @ self.weight - self.targets
        return 2 * pred @ self.samples / self.samples.shape[0]
