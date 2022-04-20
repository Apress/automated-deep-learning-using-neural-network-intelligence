from numpy import exp, sqrt, cos, pi, e
from ch3.bbf.utils import noise


def ackley_function(x, y):
    """
    Ackley’s function
    """
    z = 20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) -\
        exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
    r = z + noise(x, y, scale = 4)
    return r


from ch3.bbf.utils import scatter_plot

if __name__ == '__main__':
    scatter_plot(ackley_function, [-10, 10], [-10, 10])
