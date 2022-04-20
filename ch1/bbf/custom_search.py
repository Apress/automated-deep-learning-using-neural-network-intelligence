import random
from ch1.bbf.black_box_function import black_box_function

seed = 0
random.seed(0)
max_ = -100
best_trial = None
for _ in range(100):
    x = random.choice([50, 49, 48])
    y = random.choice([2, 4, 6, 8, 10])
    z = round(random.uniform(1, 10_000), 2)
    r = black_box_function(x, y, z)
    if r > max_:
        max_ = r
        best_trial = f'(x={x}, y={y}, z={z}) -> {r}'

print(best_trial)
