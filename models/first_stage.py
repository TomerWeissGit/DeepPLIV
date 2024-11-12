
import torch.nn as nn
import torch.nn.functional as functional


class NeuralNetworkFirstStage(nn.Module):
    def __init__(self, input_dim: int):
        super(NeuralNetworkFirstStage, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        """
        Forward pass of the neural network.
        :param x: torch.Tensor, the input tensor.
        :return: torch.Tensor, the output tensor.
        """
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        x = self.fc4(x)  # Linear activation for the final layer
        return x

