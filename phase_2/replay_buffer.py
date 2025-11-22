# phase_2/replay_buffer.py

import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def store(self, state, action, reward, next_state, done):
        """Stores one transition in the buffer."""
        transition = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Samples a minibatch."""
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)
