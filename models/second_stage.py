import torch
import torch.nn as nn


class NeuralNetworkSecondStage(nn.Module):
    """
    A neural network model for the second stage of the DeepPLIV model.
    """

    def __init__(self, input_dim_1, input_dim_2):
        """
        Initialize the neural network model.
        :param input_dim_1: int, the dimension of the first input.
        :param input_dim_2: int, the dimension of the second input.
        """
        super(NeuralNetworkSecondStage, self).__init__()

        # First network (deep neural network) for the first input
        self.deep_net = nn.Sequential(
            nn.Linear(input_dim_1, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU()
        )

        # Second network (single layer) for the second input
        self.linear_net = nn.Linear(input_dim_2, 1)

        # Final layer to combine both outputs
        self.final_layer = nn.Linear(9, 1)  # 8 from each branch, resulting in a concatenated vector of size 16

    def forward(self, input1, input2):
        # Pass first input through the deep neural network
        x1 = self.deep_net(input1)

        # Pass second input through the single linear layer
        x2 = self.linear_net(input2)

        # Concatenate both processed inputs
        x = torch.cat((x1, x2), dim=1)

        # Pass the concatenated result through the final layer
        output = self.final_layer(x)
        return output
