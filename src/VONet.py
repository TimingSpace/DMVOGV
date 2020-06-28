# This code is based on https://github.com/ClementPinard/SfmLearner-Pytorch
import torch
import torch.nn as nn


def conv(in_planes, out_planes, kernel_size=3,stride=2,dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=0, stride=stride,dilation=dilation),

        #nn.SyncBatchNorm(out_planes),
        nn.GroupNorm(8,out_planes),
        #nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )
def dia_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2,stride=2,dilation = 2),

        #nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )
## Current
class SPADVONet(nn.Module):

    def __init__(self,coor_layer_flag=True,color_flag=True):
        super(SPADVONet, self).__init__()

        input_channel = 6
        if color_flag == False:
            input_channel =2
        if coor_layer_flag:
            input_channel+=2


        conv_planes = [input_channel,16, 32, 64, 2]
        kernel_sizes= [(3,9), (3,9),  (3,7), (3,7)]     
        dilations   = [2,  2,  2, 2]     
        #base
        layers = []
        feature_layer = 3
        for i in range(0,feature_layer):
            layers.append(conv(conv_planes[i], conv_planes[i+1], kernel_size=kernel_sizes[i],  dilation=dilations[i]))

        #vo
        vo_pred = nn.Conv2d(conv_planes[-2], conv_planes[-1], kernel_size=kernel_sizes[3],dilation = dilations[-1], padding=0)

        self.vonet=nn.Sequential(*layers,vo_pred)


    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTransvo2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_pairs):
        input = image_pairs
        vo  = self.vonet(input)
        return vo.mean([2,3])

