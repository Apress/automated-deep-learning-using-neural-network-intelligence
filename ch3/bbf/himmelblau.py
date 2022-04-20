from ch3.bbf.utils import scatter_plot, discrete, noise


def himmelblau_function(x, y):
    """
    Himmelblau’s function
    """
    z = -((x**2 + y - 11)**2 + (x + y**2 - 7)**2) / 4 + 2000
    d = discrete(z, .005)
    r = d + noise(x, y, scale = 500)
    return r


if __name__ == '__main__':
    scatter_plot(himmelblau_function, [-10, 10], [-10, 10])
