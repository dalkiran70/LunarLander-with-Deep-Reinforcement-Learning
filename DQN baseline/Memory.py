import torch
import numpy as np


class ExperienceReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.states = torch.zeros((buffer_size, state_dim))
        self.next_states = torch.zeros((buffer_size, state_dim))
        self.dones = torch.zeros(buffer_size)
        self.actions = torch.zeros(buffer_size)
        self.rewards = torch.zeros(buffer_size)
        self.memory_pointer = 0
        self.first_fill = 0
        self.fill_flag = False

    def sample_batch(self):
        index = np.random.choice(self.first_fill, self.batch_size, replace=False)
        s = self.states[index]
        next_s = self.next_states[index]
        a = self.actions[index]
        d = self.dones[index]
        r = self.rewards[index]
        return s, a, r, next_s, d

    def store(self, state, action, reward, next_state, done):
        if self.memory_pointer == self.buffer_size:
            self.fill_flag = True
            self.memory_pointer = 0
        self.states[self.memory_pointer] = state
        self.actions[self.memory_pointer] = action
        self.rewards[self.memory_pointer] = reward
        self.next_states[self.memory_pointer] = next_state
        self.dones[self.memory_pointer] = done
        self.memory_pointer += 1
        if not self.fill_flag: self.first_fill += 1
        return
