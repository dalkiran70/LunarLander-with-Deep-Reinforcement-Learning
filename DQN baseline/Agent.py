from Q_Network import DQN
from Memory import ExperienceReplayBuffer
import numpy as np
import torch
import copy


class Agent:
    def __init__(self, epsilon, input_dim, hidden1, hidden2, output_dim, buffer_size, batch_size, gamma, tau):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.Q_network = DQN(input_dim, hidden1, hidden2, output_dim)
        self.target_Q = copy.deepcopy(self.Q_network)
        self.memory = ExperienceReplayBuffer(buffer_size, batch_size, input_dim, output_dim)


    def select_action(self, state):
        """Does return a single number"""
        number = np.random.randn(1)
        if number < 1 - self.epsilon:
            action = torch.argmax(self.Q_network.forward(state))
        else:
            action = torch.randint(0, self.output_dim, (1,))
        return action

    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
        return

    def update(self):
        span = range(self.batch_size)
        states, actions, rewards, next_states, dones = self.memory.sample_batch()
        Q_vals = self.Q_network(states)[span, actions[span].long()]
        max_Q = torch.max(self.Q_network(next_states), dim=1)[0]
        with torch.no_grad():
            Q_bootstrap = max_Q * self.gamma * (1 - dones) + rewards
        # backpropagation
        self.Q_network.optimizer.zero_grad()
        loss = torch.nn.MSELoss()
        error = loss(Q_vals, Q_bootstrap)
        error.backward()
        self.Q_network.optimizer.step()

        for target_param, param in zip(self.target_Q.parameters(), self.Q_network.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return