"""
Simple VO Net module
"""

# This code is based on https://github.com/ClementPinard/SfmLearner-Pytorch
import torch
import torch.nn as nn


def conv(in_planes, out_planes, kernel_size=3,stride=2,dilation=1):
    """
    Basic convolutional block, including a 2d convolution layer, group normalization layer and  rectified linear unit layers
    Attributes:
        in_planes  : channel number of the input feature map, (type: int;  value >0 )
        out_planes : channel number of the output feature map, which indicate the convolutional kernel number, 
            because each kernal generate a feature, (type: int; value >0 )
        kernel_size: the size of each convolutional kernel, can be a number like 3 or tuple like (3,5), (3, 5) means the kernal size 
            is 3x5 and 3 is the row number, 5 is the column number; 3 means that the kernel size is 3x3; (type: int or tuple; value >0)
        stride     : stride step size of convolutional layer, (type: int or tuples; value >0)
        dialation  : dilation size of convolutional layer, (type: int or tuples; value >0)
    Return:
        A basic convolutional block, including a 2d convolution layer, group normalization layer and  rectified linear unit layers
    Raise:
        NO
    """
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=0, stride=stride, dilation=dilation), # convlutional layer
        nn.GroupNorm(8,out_planes), # group normalization layer
        nn.ReLU(inplace=True)       # recified linear unit layer
    )


class SPADVONet(nn.Module):
    """
    Simple Patch Agreement Deep Visual Odometry Model， this module inherit from base module named nn.Module
    """
    def __init__(self, coor_layer_flag=True, color_flag=True, vo_dimension = 2):
        """
        SPADVONet initialization function, this function is called when you run pad_model = SPADVONet(coor_layer_flag, color_flag)
        Attribute:
            coor_layer_flag: a boolean flag indicate whether the input of this module is with coor layer or not (type: bool)
            color_flag     : a boolean flag indicate whether the input of this module is 3-channel color image or one channel gray image (type: bool)
            vo_dimension   : indicate the  dimension of output vo, can be 1(only for scale), 2 (scale and rotation) 3 (x, y and rotation) and 6(6 dof motion)
        Return:
            NO
        Raise: 
            NO
        """
        super(SPADVONet, self).__init__() # intialize the base module (nn.Module)

        input_channel = 6                 # set default input channel as 6 (stacked two rgb color images)
        if color_flag == False:           
            input_channel = 2             # if the color flag is set False, set the input channel as 2 (stacked two gray image)
        if coor_layer_flag:
            input_channel += 2            # if coor layer flag is set True, two more channel should be added (x-coor, y-coor)
        
        self.input_channel = input_channel
        # config the parameter of different layers
        conv_planes = [input_channel, 16, 32, 64, vo_dimension] 
        kernel_sizes= [(3,9), (3,9),  (3,7), (3,7)]     
        dilations   = [2,  2,  2, 2]     

        # construct layers 
        layers = []
        feature_layer = 3   # feature layer number
        for i in range(0, feature_layer):
            layers.append(conv(conv_planes[i], conv_planes[i+1], kernel_size=kernel_sizes[i], dilation=dilations[i]))

        # construct vo prediction layer (also convoltuion layer)
        vo_pred = nn.Conv2d(conv_planes[-2], conv_planes[-1], kernel_size=kernel_sizes[3],dilation = dilations[-1], padding=0)

        self.vonet=nn.Sequential(*layers,vo_pred)


    def init_weights(self):
        """
        Intialize the parameter of the convolutional layer
        Attributes: No
        Returns: No
        Raise: No
        """ 
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTransvo2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, image_pairs):
        """
        Forward function of this module, it is called when you run vo = pad_model(image_pairs)
        Attribute:
            image_pairs: input stacked image pairs, shape [B, C, H, W] c should be image to self.input_channel, type: tensor of float
        Return:
            prediected vo result 
        """
        assert len(image_pairs.shape) == 4, 'wrong input shape, should be [B, C, H, W], but get {}'.format(image_pairs.shape)
        assert image_pairs.shape[1]   == self.input_channel, 'wrong input dimension, should be same with self.input channel, but get input channel {}, self.input channel {}'. format(image_pairs.shape[1], self.input_channel)
        
        input = image_pairs
        vo  = self.vonet(input)
        return vo.mean([2,3]) #[B, self.vo_dimensions]

# module test functions 

if __name__ == '__main__':
    # model construction
    vo_model = SPADVONet(coor_layer_flag=True, color_flag=False, vo_dimension=1)

    # test data generation
    image_pairs = torch.rand((10, 4, 480, 640))
    vo_gt        = torch.rand((10, 1))
    # forward vo

    vo = vo_model(image_pairs)

    # calculate loss 
    loss = torch.mean(torch.sum(vo-vo_gt))

    # backward
    loss.backward()

    print('vo model test past')

