from random import random


class DummyModel:

    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def train(self):
        # Training here
        ...

    def test(self):
        # Test results
        return round(self.x + self.y + random() / 10, 2)
