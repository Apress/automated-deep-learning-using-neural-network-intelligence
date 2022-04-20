from numpy import exp, sqrt, cos, sin, pi


def holder_function(x, y):
    """
    Holder’s function
    """
    z = abs(sin(x) * cos(y) * exp(abs(1 - (sqrt(x**2 + y**2) / pi))))
    d = discrete(z, .8)
    r = d + noise(x, y, scale = 8)
    d = discrete(r, .2)
    return d


from ch3.bbf.utils import scatter_plot, discrete, noise

if __name__ == '__main__':
    scatter_plot(holder_function, [-10, 10], [-10, 10])
