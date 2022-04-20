from numpy import arange
from pylab import meshgrid, cm
import matplotlib.pyplot as plt
import numpy as np


def discrete(y, eta):
    """
    Function discretization
    """
    if isinstance(y, np.ndarray):
        return (y * eta).astype(np.int).astype(np.double) / eta
    else:
        return round(y * eta) / eta


def noise(x, y, scale = 1):
    """
    Random noise addition
    """
    z = np.sin(x) * np.cos(y)
    return z * scale


def scatter_plot(f, x_range, y_range, points = None, title = None):
    if points is None:
        points = []

    x = arange(x_range[0], x_range[1], 0.1)
    y = arange(y_range[0], y_range[1], 0.1)
    X, Y = meshgrid(x, y)
    Z = f(X, Y)

    im = plt.imshow(
        Z,
        cmap = cm.bwr,
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    )
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])

    plt.scatter(
        [p['x'] for p in points],
        [p['y'] for p in points],
        color = 'black'
    )

    if title:
        plt.title(title)

    plt.show()
