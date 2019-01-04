
import random


class Memory:

    def __init__(self, size=5000):
        self.memory = []
        self.max_memory = size

    def add(self, experience):
        self.memory.append(experience)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def sample(self, size=32):
        return random.sample(self.memory, min(len(self.memory), size))
