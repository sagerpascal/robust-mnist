import re

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, accuracy_score, precision_score


class Meter:
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Resets the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass


class AverageValueMeter(Meter):
    """ Meter to calculate a running average """

    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Activation(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


class BaseMetric(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name

    def to_numpy(self, x):
        return x.detach().numpy()


class Accuracy(BaseMetric):

    def __init__(self, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return accuracy_score(self.to_numpy(y_gt), self.to_numpy(y_pr))


class Recall(BaseMetric):

    def __init__(self, average='weighted', activation=None, **kwargs):
        super().__init__(**kwargs)
        self.average = average
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return recall_score(self.to_numpy(y_gt), self.to_numpy(y_pr), average=self.average, labels=np.unique(np.concatenate((y_pr, y_gt), axis=0)))


class Precision(BaseMetric):

    def __init__(self, average='weighted', activation=None, **kwargs):
        super().__init__(**kwargs)
        self.average = average
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return precision_score(self.to_numpy(y_gt), self.to_numpy(y_pr), average=self.average, labels=np.unique(np.concatenate((y_pr, y_gt), axis=0)))
