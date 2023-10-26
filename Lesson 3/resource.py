import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(model=None,
                         x_lim=None, y_lim=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)

    learning_path = model.loss_dict
    plt.plot(learning_path.keys(), learning_path.values())
    plt.title(f'Кривая обучения {model.__class__.__name__}')
    if x_lim or y_lim:
        plt.xlim(x_lim[0], x_lim[1])
        plt.ylim(y_lim[0], y_lim[1])

    ax.set_xlabel('Номер итерации')
    ax.set_ylabel('Функция потерь')
    ax.legend(['Loss: {}, {} итераций'.format(round(list(learning_path.values())[-1], ndigits=2),
                                              list(learning_path.keys())[-1])])
    plt.show()


def plot_weight_curves(model=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)

    learning_path = model.weight_dict
    data = list(learning_path.values())

    first_w0 = [item[1] for item in data]
    second_w1 = [item[2] for item in data]

    plt.title(f'$w_1$, $w_2$ {model.__class__.__name__}')
    plt.xlabel(r'$w_1$')
    plt.ylabel(r'$w_2$')

    plt.plot(first_w0, second_w1, 'o-', markersize=5)
    plt.show()


def plot_weights(weights):
    numbers = np.arange(0, len(weights))
    tick_labels = ['w' + str(num) for num in numbers]
    cc = [''] * len(numbers)
    for n, val in enumerate(weights):
        if val < 0:
            cc[n] = 'red'
        elif val >= 0:
            cc[n] = 'blue'

    plt.bar(x=numbers, height=weights, color=cc)
    plt.xticks(np.arange(0, len(weights)), tick_labels)


class MyLinearRegression:
    def __init__(self, samples: pd.DataFrame, targets: pd.DataFrame,
                 fit_intercept: bool = True, copy: bool = True):
        self.weight = None
        self.fit_intercept = fit_intercept
        self.samples = samples.copy() if copy else samples
        self.targets = targets

    def fit(self):
        if self.fit_intercept:
            self.samples = np.hstack((self.samples, np.ones((self.samples.shape[0], 1))))

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
                 min_los: float = 1.0,
                 print_cost: bool = False,
                 random_state=42,
                 **kwargs):
        super().__init__(**kwargs)
        self.iters = iters
        self.alpha = alpha
        self.diff_mse = diff_mse
        self.min_los = min_los
        self.seed = random_state
        self.print_cost = print_cost
        self.loss_dict = {}
        self.weight_dict = {0: np.zeros(3)}

    def init(self, weights_size):
        np.random.seed(self.seed)
        return np.random.randn(weights_size) / np.sqrt(weights_size)

    def loss(self):
        yhat = self.predict()
        return np.square(yhat - self.targets).sum() / self.targets.size

    def update(self):
        return self.weight - self.alpha * self._calc_gradient()

    def fit(self):

        if self.weight is None:  # если веса не заданы - задаем
            self.weight = self.init(self.samples.shape[1])

        if self.fit_intercept:  # если задано смещение - задаем
            self.weight = np.hstack((self.init(1), self.weight))
            self.samples = np.hstack((np.ones((self.samples.shape[0], 1)), self.samples))

        previous_cost = self.loss()

        self.loss_dict[0] = previous_cost
        for count in range(1, self.iters + 1):

            self.weight = self.update()
            current_cost = self.loss()

            if count % 100 == 0 and self.print_cost is True:
                print(f"Cost at iteration {count} is {current_cost}, weight={self.weight}")

            self.loss_dict[count] = current_cost
            self.weight_dict[count] = self.weight

            if np.abs(current_cost - previous_cost) < self.diff_mse or self.loss() < self.min_los:
                print(
                    f'Model alpha: {self.alpha}, diff_mse: {self.diff_mse}, iterations: {count}, loss: {self.loss()} ...')
                break

            previous_cost = current_cost

    def _calc_gradient(self):
        yhat = self.predict()
        return 2 * (yhat - self.targets) @ self.samples / self.samples.shape[0]


class MinMaxScaler:
    def __init__(self):
        self.min = 0
        self.max = 0

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class StandardScaler:
    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std + self.mean

    def inverse_transform_coef(self, coef_scaled):
        return (coef_scaled / self.std).iloc[0]

    def inverse_transform_intercept(self, intercept_scaled, coef_scaled):
        return intercept_scaled - np.sum(coef_scaled * self.mean / self.std)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LassoGradientLinearRegression(MyGradientLinearRegression):
    def __init__(self,
                 l1_penalty=0.001,
                 **kwargs):
        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty

    def loss(self):
        yhat = self.predict()
        l1_term = self.l1_penalty * np.sum(np.abs(self.weight[1:]))
        return np.square(yhat - self.targets).mean() + l1_term

    def update(self):
        return self.weight - self.alpha * (self._calc_gradient() + np.sign(self.weight) * self.l1_penalty)


class RidgeGradientLinearRegression(MyGradientLinearRegression):
    def __init__(self,
                 l2_penalty=0.001,
                 **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty

    def loss(self):
        yhat = self.predict()
        l2_term = (self.l2_penalty / 2) * np.sum(np.square(self.weight[1:]))
        return np.square(yhat - self.targets).mean() + l2_term

    def update(self):
        l2_term = self.l2_penalty * np.mean(self.weight[1:])
        return self.weight - self.alpha * (self._calc_gradient() + l2_term)


class ElasticGradientLinearRegression(MyGradientLinearRegression):
    def __init__(self,
                 l1_penalty=0.0,
                 l2_penalty=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

    def loss(self):
        yhat = self.predict()
        l1_term = self.l1_penalty * np.sum(np.abs(self.weight[1:]))
        l2_term = (self.l2_penalty / 2) * np.sum(np.square(self.weight[1:]))
        return np.square(yhat - self.targets).mean() + l1_term + l2_term

    def update(self):
        l2_term = self.l2_penalty * np.sum(self.weight[1:])
        return self.weight - self.alpha * (self._calc_gradient() + np.sign(self.weight) * self.l1_penalty + l2_term)
