import torch
import torch.nn as nn
import torch.nn.functional as F
counting = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TaikoQFunction(nn.Module):
    def __init__(self,width:int=80,height:int=19,channels=3):
        """输入RGB图像(3通道)，输出3个动作空间的分布概率"""
        super(TaikoQFunction,self).__init__()
        self.channels = channels
        self.action_dim = 3

        self.conv1 = nn.Conv2d(self.channels,16,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(16,16,kernel_size=3,stride=1)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(
            conv2d_size_out(width, kernel_size=8, stride=4), kernel_size=3, stride=1)

        convh = conv2d_size_out(
            conv2d_size_out(height, kernel_size=8, stride=4), kernel_size=3, stride=1)

        self.fc3 = nn.Linear(convh * convw*16, 16)
        self.fc4 = nn.Linear(16, self.action_dim)

    def forward(self,state):
        x = F.leaky_relu(self.conv2(F.leaky_relu(self.conv1(state))))
        x = x.view( -1,272)
        return self.fc4( F.leaky_relu(self.fc3(x)))



