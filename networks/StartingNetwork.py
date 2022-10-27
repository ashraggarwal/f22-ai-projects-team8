import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """

    def __init__(self,dimension):
        super().__init__()
        self.fc1 = nn.Linear(dimension, 100) # What could that number mean!?!?!? Ask an officer to find out :)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x (tensor): the input to the model
        '''
        x = nn.functional.relu(self.fc1(x.squeeze(1).float()))
        x = nn.functional.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))


