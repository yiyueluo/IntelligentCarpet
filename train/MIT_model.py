import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils_func import softmax

def softmax(data):
    for i in range(data.shape[0]):
        f = data[i,:].reshape (data.shape[1])
        data[i,:] = torch.exp(f) / torch.sum(torch.exp(f))
    return data

class tile2openpose_conv3d(nn.Module):
    def __init__(self, windowSize):
        super(tile2openpose_conv3d, self).__init__()   #tactile 64*64
        if windowSize == 0:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
        else:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(2*windowSize, 32, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
        # 32*64*64

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)) # 48 * 48
        # 64*32*32

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        #128*32*32

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)) # 24 * 24
        #256*16*16

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))
        #512*16*16

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(5,5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024))
        #1024*8*8

        self.conv_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2)) # 10 * 10
        #1024*4*4

        self.l1 = nn.Sequential(
            nn.Linear(1024*2*2, 512), #for 32*32: nn.Linear(1024*2*2, 512)
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.l2 = nn.Sequential(
            nn.Linear(512, 2),
            nn.Dropout(0.5)
        )



    def forward(self, input, device):
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)
        output = self.conv_5(output)
        output = self.conv_6(output)

        output = output.view(input.shape[0], 1024*2*2)
        output = self.l1(output)
        output = self.l2(output)


        # output = output.reshape(output.shape[0],output.shape[1],output.shape[2],output.shape[3],1)
        # output = output.repeat(1,1,1,1,9)
        #
        # layer = torch.zeros(output.shape[0], 1, output.shape[2], output.shape[3], output.shape[4]).to(device)
        # for i in range(layer.shape[4]):
        #     layer[:,:,:,:,i] = i
        # layer = layer/(layer.shape[4]-1)
        # output = torch.cat( (output,layer), axis=1)

        # print (output.shape)

        return output


