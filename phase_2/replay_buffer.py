import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def store(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, bs):
        if len(self.buffer) < bs:
            return None
        batch = random.sample(self.buffer, bs)
        return map(np.array, zip(*batch))

    def size(self):
        return len(self.buffer)
