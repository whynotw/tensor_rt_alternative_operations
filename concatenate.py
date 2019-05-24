import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # define conv2d and initialize weights with 0 and 1
        self.conv2d1 = nn.Conv2d(in_channels=channels,
                                 out_channels=channels*2,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 groups=1,
                                 bias=False,
                                 dilation=1)
        filter1_weight = np.zeros((channels*2,channels,1,1),dtype=np.float32)
        for c in range(channels):
            filter1_weight[c,c,:,:] = 1.
        self.conv2d1.weight.data = torch.Tensor(filter1_weight)

        self.conv2d2 = nn.Conv2d(in_channels=channels,
                                 out_channels=channels*2,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 groups=1,
                                 bias=False,
                                 dilation=1)
        filter2_weight = np.zeros((channels*2,channels,1,1),dtype=np.float32)
        for c in range(channels):
            filter2_weight[c+channels,c,:,:] = 1.
        self.conv2d2.weight.data = torch.Tensor(filter2_weight)

    # upsampling method in https://github.com/ultralytics/yolov3/blob/master/models.py
    def forward(self, images_input):
        return torch.cat(images_input,1)

    # upsampling method by conv2d
    def forward_alt(self, images_input):
        image1_output = self.conv2d1(images_input[0])
        image2_output = self.conv2d2(images_input[1])
        return image1_output+image2_output

if __name__ == "__main__":
    # settings
    image_size = 3
    channels = 2
    batch = 1

    # create two images with shape (batch,channels,image_size,image_size)
    image1_input = np.array( 
                   np.arange(batch*channels*image_size*image_size)
                   .reshape((batch,channels,image_size,image_size)),
                   dtype=np.float32)
    image2_input = np.array( 
                  -np.arange(batch*channels*image_size*image_size)
                   .reshape((batch,channels,image_size,image_size)),
                   dtype=np.float32)

    print("input shape is ", end="")
    print(image1_input.shape)
    print(image1_input)
    print(image2_input)
    print("-"*50)

    # upsample this image to shape (batch,channels,size*2,size*2)
    # by copying each pixel from 1x1 to 2x2 square
    # two methods are implemented
    net = Net()

    images_input = [torch.from_numpy(image1_input),torch.from_numpy(image2_input)]

    images_output = net.forward(images_input)
    print("output shape is ", end="")
    print(tuple(images_output.size()))
    print(images_output.detach().numpy())
    print("-"*50)

    images_output = net.forward_alt(images_input)
    print("output shape is ", end="")
    print(tuple(images_output.size()))
    print(images_output.detach().numpy())
