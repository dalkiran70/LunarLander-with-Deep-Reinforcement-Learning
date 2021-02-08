import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, layer1_dims, layer2_dims, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, layer1_dims)
        self.fc2 = nn.Linear(layer1_dims, layer2_dims)
        self.act_lay = nn.Linear(layer2_dims, action_dim)

    def forward(self, obs):
        out = F.relu(self.fc1(obs))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.act_lay(out))
        return out


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, layer1_dims, layer2_dims, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, layer1_dims)
        self.fc2 = nn.Linear(layer1_dims + action_dim, layer2_dims)
        self.q_lay = nn.Linear(layer2_dims, 1)

    def forward(self, obs, action):
        out = F.relu(self.fc1(obs))
        out = F.relu(self.fc2(torch.cat((out, action), dim=1)))
        out = self.q_lay(out)

        return out
