from ch3.bbf.utils import discrete, noise, scatter_plot


def black_box_f1(x, y):
    z = - 10 * (pow(x, 5) / (3 * pow(x * x * x / 4 + 1, 2) + pow(y, 4) + 10) + pow(x * y / 2, 2) / 1000)
    d = discrete(z, .8)
    r = d + noise(x, y, scale = 8)
    d = discrete(r, .2)
    return d


if __name__ == '__main__':
    scatter_plot(black_box_f1, [-10, 10], [-10, 10])
