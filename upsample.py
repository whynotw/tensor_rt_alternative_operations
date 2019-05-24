import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # define convTranspose2d and initialize weights with 0 and 1
        self.convtrans = nn.ConvTranspose2d(in_channels=channels,
                                            out_channels=channels,
                                            kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            output_padding=0,
                                            groups=1,
                                            bias=False,
                                            dilation=1)
        filter_weight = np.zeros((channels,channels,2,2),dtype=np.float32)
        for c in range(channels):
            filter_weight[c,c,:,:] = 1.
        self.convtrans.weight.data = torch.Tensor(filter_weight)

    # upsampling method in https://github.com/ultralytics/yolov3/blob/master/models.py
    def forward(self, image_input):
        return F.interpolate(image_input, scale_factor=2, mode='nearest')

    # upsampling method by convTranspose2d
    def forward_alt(self, image_input):
        return self.convtrans(image_input)

if __name__ == "__main__":
    # settings
    image_size = 3
    channels = 2
    batch = 1

    # create an image with shape (batch,channels,image_size,image_size)
    image_input = np.array( 
                  np.arange(batch*channels*image_size*image_size)
                  .reshape((batch,channels,image_size,image_size)),
                  dtype=np.float32)

    print("input shape is ", end="")
    print(image_input.shape)
    print(image_input)
    print("-"*50)

    # upsample this image to shape (batch,channels,size*2,size*2)
    # by copying each pixel from 1x1 to 2x2 square
    # two methods are implemented
    net = Net()

    image_output0 = net.forward(torch.from_numpy(image_input))
    print("output shape is ", end="")
    print(tuple(image_output0.size()))
    print(image_output0.detach().numpy())
    print("-"*50)

    image_output1 = net.forward_alt(torch.from_numpy(image_input))
    print("output shape is ", end="")
    print(tuple(image_output1.size()))
    print(image_output1.detach().numpy())
