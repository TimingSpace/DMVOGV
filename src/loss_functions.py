from torch import nn
from torch.autograd import Variable
import torch
import numpy as np

celoss = nn.CrossEntropyLoss()

# predict_result b c h w
# ground truth   b c 1 1
# attention      b 1 h w


def SingleShotAttentionLoss(predict_result,ground_truth,attention,mask=[0,1,2,3,4,5]):
    diff = ground_truth[:,mask]-predict_result[:,mask]
    diff_s = diff.pow(2)
    #loss = diff_s*torch.exp(attention)+0.1*torch.abs(attention)
    loss = diff_s*torch.exp(attention)+0.1*attention.pow(2)
    loss = loss/(loss.size()[0]*loss.size()[2]*loss.size()[3])
    loss = loss.sum()
    return loss
def WeightedMSELoss(predict_result,ground_truth,weights=torch.FloatTensor([1,1,1,100,100,100])):
    diff = ground_truth-predict_result
    diff_s = diff.pow(2)
    loss = diff_s*weights
    loss = loss/loss.size()[0]
    loss = loss.sum()
    return loss
# ground vehicle loss
def GVLoss(predict_result,ground_truth):
    diff = ground_truth[:,[2,4]]-predict_result
    #diff = ground_truth-predict_result
    diff_s = diff.pow(2)
    loss = diff_s
    loss = loss/loss.size()[0]
    loss = loss.sum()
    return loss

def MSELoss(predict_result,ground_truth,mask=[0,1,2,3,4,5]):
    diff = ground_truth[:,mask]-predict_result[:,mask]
    #diff = ground_truth-predict_result
    diff_s = diff.pow(2)
    loss = diff_s
    loss = loss/loss.size()[0]
    loss = loss.sum()
    return loss
def ReliabilityMetric(predict_result,ground_truth,attention):
    ground_truth = ground_truth.view(ground_truth.size(0),6,1,1)
    diff = ground_truth-predict_result
    diff_s = diff.pow(2)
    loss = diff_s.sum(1).view(ground_truth.size(0),-1)
    loss /= torch.sum(loss)
    attention = attention.view(ground_truth.size(0),-1)
    attention_exp = -attention*torch.exp(-attention)
    attention_exp/=torch.sum(attention_exp)
    div= torch.nn.functional.kl_div(torch.log(attention_exp),loss)
    return div

def PatchLoss(predict_result,ground_truth):
    ground_truth = ground_truth.view(ground_truth.size(0),6,1,1)
    diff = ground_truth-predict_result
    diff_s = diff.pow(2)
    loss = diff_s.sum(1)
    return loss

def SingleShotLoss(predict_result,ground_truth,mask=[0,1,2,3,4,5]):
    diff = ground_truth[:,mask]-predict_result[:,mask]
    diff_s = diff.pow(2)
    loss = diff_s
    loss = loss/(loss.size()[0]*loss.size()[2]*loss.size()[3])
    loss = loss.sum()
    return loss
def GroupWithATTLoss(f_12,f_g_12,att_12,b_21,b_g_21,att_21,f_23,f_g_23,att_23,b_32,b_g_32,att_32,f_13,f_g_13,att_13,b_31,b_g_31,att_31,mask=[0,1,2,3,4,5]):
    f_g_12 = f_g_12.view(f_g_12.size(0),6,1,1)
    #f_g_13 = f_g_13.view(f_g_12.size(0),6,1,1)
    #f_g_23 = f_g_23.view(f_g_12.size(0),6,1,1)
    b_g_21 = b_g_21.view(f_g_12.size(0),6,1,1)
    #b_g_32 = b_g_32.view(f_g_12.size(0),6,1,1)
    #b_g_31 = b_g_31.view(f_g_12.size(0),6,1,1)
    f_12_loss = SingleShotAttentionLoss(f_12,f_g_12,att_12,mask)
    #f_23_loss = SingleShotAttentionLoss(f_23,f_g_23,att_23)
    #f_13_loss = SingleShotAttentionLoss(f_13,f_g_13,att_13)
    b_21_loss = SingleShotAttentionLoss(b_21,b_g_21,att_21,mask)
    #b_32_loss = SingleShotAttentionLoss(b_32,b_g_32,att_32)
    #b_31_loss = SingleShotAttentionLoss(b_31,b_g_31,att_31)
    loss = f_12_loss+b_21_loss
    return loss

def ScaleLoss(f_12,g_12):
    scale_g = torch.sqrt(torch.sum(g_12[:,0:3]*g_12[:,0:3],1).view(g_12.shape[0],1))
    diff = scale_g - f_12 
    diff_s = diff.pow(2)
    loss = diff_s
    loss = loss.mean()
    return loss

def GroupScaleLoss(f_12,f_g_12,b_21,b_g_21):
    f_12 = f_12.mean(3).mean(2)
    b_21 = b_21.mean(3).mean(2)
    f_12_loss = ScaleLoss(f_12,f_g_12)
    b_21_loss = ScaleLoss(b_21,b_g_21)
    loss = f_12_loss+b_21_loss

    return loss

def GroupGVLoss(f_12,f_g_12,b_21,b_g_21):
    f_12_loss = GVLoss(f_12,f_g_12)
    b_21_loss = GVLoss(b_21,b_g_21)
    loss = f_12_loss+b_21_loss
    return loss



def GroupLoss(f_12,f_g_12,b_21,b_g_21,f_23,f_g_23,b_32,b_g_32,f_13,f_g_13,b_31,b_g_31,mask=[0,1,2,3,4,5]):
    f_12 = f_12.mean(3).mean(2)
    #f_13 = f_13.mean(3).mean(2)
    #f_23 = f_23.mean(3).mean(2)
    b_21 = b_21.mean(3).mean(2)
    #b_32 = b_32.mean(3).mean(2)
    #b_31 = b_31.mean(3).mean(2)
    f_12_loss = MSELoss(f_12,f_g_12,mask)
    #f_23_loss = MSELoss(f_23,f_g_23)
    #f_13_loss = MSELoss(f_13,f_g_13)
    b_21_loss = MSELoss(b_21,b_g_21,mask)
    #b_32_loss = MSELoss(b_32,b_g_32)
    #b_31_loss = MSELoss(b_31,b_g_31)
    loss = f_12_loss+b_21_loss

    return loss

def disc_loss(result,groundtruth):
    r = result[0]
    t = result[1]
    #print(r.shape,t.shape,groundtruth.shape)
    r = r.mean((2,3))
    r_diff = groundtruth[:,4]-r
    #diff = ground_truth-predict_result
    diff_s = r_diff.pow(2)
    loss = diff_s
    loss = loss/loss.size()[0]
    loss_r = loss.sum()
    #print(r.shape,t.shape,groundtruth.shape)
    #print(loss_r)
    z = groundtruth[:,2]
    #z  +=  0.05
    #z  *=  10
    z   +=  0.4
    z   = z.long()
    #print(z)
    loss_t = celoss(t,z)
    #print(loss_t)
    return loss_t



def GroupWithSSLoss(f_12,f_g_12,b_21,b_g_21,f_23,f_g_23,b_32,b_g_32,f_13,f_g_13,b_31,b_g_31,mask=[0,1,2,3,4,5]):
    f_g_12 = f_g_12.view(f_g_12.size(0),6,1,1)
    #f_g_13 = f_g_13.view(f_g_12.size(0),6,1,1)
    #f_g_23 = f_g_23.view(f_g_12.size(0),6,1,1)
    b_g_21 = b_g_21.view(f_g_12.size(0),6,1,1)
    #b_g_32 = b_g_32.view(f_g_12.size(0),6,1,1)
    #b_g_31 = b_g_31.view(f_g_12.size(0),6,1,1)
    f_12_loss = SingleShotLoss(f_12,f_g_12,mask)
    #f_23_loss = SingleShotLoss(f_23,f_g_23)
    #f_13_loss = SingleShotLoss(f_13,f_g_13)
    b_21_loss = SingleShotLoss(b_21,b_g_21,mask)
    #b_32_loss = SingleShotLoss(b_32,b_g_32)
    #b_31_loss = SingleShotLoss(b_31,b_g_31)
    loss = f_12_loss+b_21_loss

    return loss


def GroupWithMSELoss(f_12,f_g_12,b_21,b_g_21,f_23,f_g_23,b_32,b_g_32,f_13,f_g_13,b_31,b_g_31,weights=torch.FloatTensor([1,1,1,100,100,100])):
    f_12_loss = WeightedMSELoss(f_12,f_g_12,weights)
    f_23_loss = WeightedMSELoss(f_23,f_g_23,weights)
    f_13_loss = WeightedMSELoss(f_13,f_g_13,weights)
    b_21_loss = WeightedMSELoss(b_21,b_g_21,weights)
    b_32_loss = WeightedMSELoss(b_32,b_g_32,weights)
    b_31_loss = WeightedMSELoss(b_31,b_g_31,weights)
    #loss = forward_mse_loss+backward_mse_loss+cycle_loss
    loss = f_12_loss+b_21_loss
    #loss = f_12_loss
    #loss = f_12_loss+f_23_loss+0.3*f_13_loss+b_21_loss+b_32_loss+0.3*b_31_loss

    #loss = backward_mse_loss
    #loss = forward_mse_loss
    return loss

def FullSequenceLoss(predict,groundtruth):
    diff = groundtruth - predict
    diff=diff*diff
    diff = diff/diff.shape[0]
    loss = np.sum(diff)
    return loss

if __name__ == '__main__':
    predict_result = torch.autograd.Variable(torch.FloatTensor(4,6,30,100).zero_())
    ground_truth = torch.autograd.Variable(torch.FloatTensor(4,6).zero_(),requires_grad=True)
    #loss = MSELoss(predict_result,ground_truth)
    loss = GroupLoss(predict_result,ground_truth,predict_result,ground_truth,predict_result,ground_truth,predict_result,ground_truth,predict_result,ground_truth,predict_result,ground_truth)
    print(loss)

