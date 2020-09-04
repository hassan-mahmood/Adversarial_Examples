
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)					# Conv layer - 10 kernels
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)					# Conv layer - 20 kernels
        self.conv2_drop = nn.Dropout2d()								# Dropout randomly 50%	
        self.fc1 = nn.Linear(320, 50)									# Fully connected layer 320 -> 50 neurons
        self.fc2 = nn.Linear(50, 10)									# Fully connected layer 50 -> 10 neurons (number of classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))						# Conv -> Pool -> ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))		# Conv -> Dropout -> Pool -> ReLU

        x = x.view(-1, 320)												# Flatten the vector
        x = F.relu(self.fc1(x))											# Fully connected -> ReLU
        x = F.dropout(x, training=self.training)						# Dropout 
        x = self.fc2(x)													# Logits
        return F.log_softmax(x, dim=1)									# Probs of final classes