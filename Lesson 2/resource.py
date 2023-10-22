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
    ax.set_ylabel('Среднеквадратическая ошибка')
    ax.legend(['MSE: {}, {} итераций'.format(round(list(learning_path.values())[-1], ndigits=2),
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
                 print_cost: bool = False,
                 random_state=42,
                 **kwargs):
        super().__init__(**kwargs)
        self.iters = iters
        self.alpha = alpha
        self.diff_mse = diff_mse
        self.seed = random_state
        self.print_cost = print_cost
        self.loss_dict = {}
        self.weight_dict = {0: np.zeros(3)}

    def init(self, weights_size):
        np.random.seed(self.seed)
        return np.random.randn(weights_size) / np.sqrt(weights_size)

    def mean_squared_error(self):
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
        previous_cost = self.mean_squared_error()

        self.loss_dict[0] = previous_cost
        for count in range(1, self.iters + 1):

            self.weight = self.update()
            current_cost = self.mean_squared_error()

            if count % 100 == 0 and self.print_cost is True:
                print(f"Cost at iteration {count} is {current_cost}, weight={self.weight}")

            self.loss_dict[count] = current_cost
            self.weight_dict[count] = self.weight

            # weight_dist = np.sum(np.abs(self.weight_dict[count - 1] - self.weight_dict[count]))

            if np.abs(current_cost - previous_cost) < self.diff_mse:
                print(f'Model alpha: {self.alpha}, diff_mse: {self.diff_mse}, iterations: {count} ...')
                break

            previous_cost = current_cost

    def _calc_gradient(self):
        yhat = self.predict()
        return 2 * (yhat - self.targets) @ self.samples / self.samples.shape[0]

