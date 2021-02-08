import numpy as np
import torch

class Memory():
    def __init__(self, mem_size, batch_size, pos_dim, action_dim, device):
        self.device = device
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.state_memory = np.zeros((mem_size, pos_dim))
        self.new_state_memory = np.zeros((mem_size, pos_dim))
        self.action_memory = np.zeros((mem_size, action_dim))
        self.reward_memory = np.zeros(mem_size)
        self.terminal_memory = np.zeros(mem_size)
        self.mem_index = 0
        self.mem_pointer = 0

    def store(self, state, new_state, action, reward, terminal):
        index = self.mem_index % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = int(terminal)
        self.mem_index += 1
        self.mem_pointer += 1
        if self.mem_pointer >= self.mem_size:
            self.mem_pointer = self.mem_size


    def sample(self):
        batch_index = np.random.choice(self.mem_pointer, self.batch_size, replace= False)
        state_batch = torch.tensor(self.state_memory[batch_index], dtype= torch.float32).to(self.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch_index], dtype= torch.float32).to(self.device)
        action_batch = torch.tensor(self.action_memory[batch_index], dtype= torch.float32).to(self.device)
        reward_batch = torch.tensor(self.reward_memory[batch_index], dtype= torch.float32).to(self.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch_index], dtype= torch.float32).to(self.device)

        return state_batch, new_state_batch, \
            action_batch, reward_batch, terminal_batch
