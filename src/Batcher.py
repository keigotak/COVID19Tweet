import random


class Batcher:
    def __init__(self, x, y, batch_size=4):
        self.x = x
        self.y = y
        self.current = 0
        self.batch_size = batch_size

    def get_batch(self):
        if self.current + self.batch_size < len(self.x):
            x_batch = self.x[self.current: self.current + self.batch_size]
            y_batch = self.y[self.current: self.current + self.batch_size]
        else:
            x_batch = self.x[self.current: ]
            y_batch = self.y[self.current: ]
        self.current += self.batch_size
        return x_batch, y_batch

    def is_batch_end(self):
        if self.current >= len(self.x):
            return True
        else:
            return False

    def reset(self, with_shuffle=False):
        self.current = 0

        if with_shuffle:
            self.shuffle()

    def shuffle(self):
        indexes = list(range(len(self.x)))
        random.shuffle(indexes)
        self.x = [self.x[i] for i in indexes]
        self.y = [self.y[i] for i in indexes]

