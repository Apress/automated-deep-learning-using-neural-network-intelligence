import random

def identity_with_parabolic_training(x):
    history = [
        max(round(x / 10, 2) * pow(h, .5) + random.uniform(-3, 3), 0)
        for h in range(1, 101)
    ]
    return x, history


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for x in range(0, 101, 10):
        final, history = identity_with_parabolic_training(x)
        plt.plot(history, label = str(x))

    plt.ylabel('Intermediate Result')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
