import nni
from numpy.random.mtrand import RandomState
import matplotlib.pyplot as plt

space = [
    nni.quniform(0, 100, 5, RandomState(seed))
    for seed in range(20)
]
plt.figure(figsize = (5, 1))
plt.title('quniform')
plt.plot(space, len(space) * [0], "x")
plt.yticks([])
plt.show()
