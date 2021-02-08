from Network import CriticNetwork, ActorNetwork
from Memory import Memory
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn


class Agent:
    def __init__(self, gamma, tau, actorlr, criticlr, variance, action_dim, mem_size, batch_size, pos_dim):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = Memory(mem_size=mem_size, batch_size=batch_size, pos_dim=pos_dim, action_dim=action_dim, device=self.device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.actor = ActorNetwork(pos_dim, 200, 200, action_dim).to(self.device)
        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=actorlr)
        self.actor_target = ActorNetwork(pos_dim, 200, 200, action_dim).to(self.device)
        self.state_dim = pos_dim
        self.critic = CriticNetwork(pos_dim, 200, 200, action_dim).to(self.device)
        self.critic_optim = optim.Adam(params=self.critic.parameters(), lr=criticlr)
        self.critic_target = CriticNetwork(pos_dim, 200, 200, action_dim).to(self.device)
        self.critic_criterion = nn.MSELoss()

        self.variance = variance

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def action_selection(self, pos):
        noise = torch.from_numpy(np.random.normal(0, self.variance, self.action_dim)).unsqueeze(0)
        action = self.actor.forward(torch.tensor(pos, dtype=torch.float).unsqueeze(0)) + noise
        return action.squeeze(0).detach().numpy()

    def update(self):
        state_batch, new_state_batch, action_batch, reward_batch, terminal_batch = self.memory.sample()
        Qvals = self.critic.forward(state_batch, action_batch)
        next_actions = self.actor_target.forward(new_state_batch)
        next_Q = self.critic_target.forward(new_state_batch, next_actions.detach())
        Qprime = reward_batch.unsqueeze(-1) + self.gamma * next_Q * (1 - terminal_batch.unsqueeze(-1))

        critic_loss = self.critic_criterion(Qvals, Qprime).to(self.device)

        # update networks
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        policy_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean().to(self.device)

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
