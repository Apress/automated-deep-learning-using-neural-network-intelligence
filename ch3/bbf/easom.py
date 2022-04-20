from numpy import exp, cos, pi

from ch3.bbf.utils import scatter_plot


def easom_function(x, y):
    """
    Easom’s function
    """
    x = x / 2
    y = y / 2
    return cos(x) * cos(y) * exp(-((x - pi)**2 + (y - pi)**2))


if __name__ == '__main__':
    scatter_plot(easom_function, [-10, 10], [-10, 10])
