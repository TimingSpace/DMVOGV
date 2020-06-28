from torch import nn
from torch.autograd import Variable
import torch
import numpy as np

celoss = nn.CrossEntropyLoss()

# predict_result b c h w
# ground truth   b c 1 1
# attention      b 1 h w

# ground vehicle loss
def GVLoss(predict_result,ground_truth):
    diff = ground_truth[:,[2,4]]-predict_result
    #diff = ground_truth-predict_result
    diff_s = diff.pow(2)
    loss = diff_s
    loss = loss/loss.size()[0]
    loss = loss.sum()
    return loss
def GroupGVLoss(f_12,f_g_12,b_21,b_g_21):
    f_12_loss = GVLoss(f_12,f_g_12)
    b_21_loss = GVLoss(b_21,b_g_21)
    loss = f_12_loss+b_21_loss
    return loss

if __name__ == '__main__':
    predict_result = torch.autograd.Variable(torch.FloatTensor(4,6,30,100).zero_())
    ground_truth = torch.autograd.Variable(torch.FloatTensor(4,6).zero_(),requires_grad=True)
    #loss = MSELoss(predict_result,ground_truth)
    loss = GroupLoss(predict_result,ground_truth,predict_result,ground_truth,predict_result,ground_truth,predict_result,ground_truth,predict_result,ground_truth,predict_result,ground_truth)
    print(loss)

