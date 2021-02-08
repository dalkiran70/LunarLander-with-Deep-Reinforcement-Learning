import torch

class DQN(torch.nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, output)
        self.optimizer = torch.optim.Adam(self.parameters())

        # transition vales

    def forward(self, state):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(state)))))

    def return_onehot(self, state):
        one_hot_act = torch.zeros(self.input_dim)
        one_hot_act[torch.argmax(self.forward(state))] = 1
        return one_hot_act
