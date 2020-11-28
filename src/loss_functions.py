from torch import nn
import torch

def GVLoss(predicted_result, ground_truth):
    """
    Calulate the difference between the predicted result and ground-truth
    Attribute:
        predicted_result: predicted result, shape [B, D]
        ground_truth    : ground truth, shape     [B, D]
    Return: 
        L2-distance
    """
    assert (len(predicted_result.shape) == 2) and (len(ground_truth.shape) == 2), 'wrong shape  of prediected result  or  ground truth'
    B_P, D_P = predicted_result.shape
    B_G, D_G = ground_truth.shape
    assert (B_P == B_G) and (D_P == D_G), 'the shape of predicted result and ground truth does not match, get {} and {}'.format(predicted_result.shape,ground_truth.shape)
    
    diff = ground_truth - predicted_result
    diff_s = diff.pow(2)
    loss = diff_s.sum() / B_G

    return loss


def GroupGVLoss(f_12, f_g_12, b_21, b_g_21):
    """
    Calculate the forward loss and backward loss
    Attribute:
        f_12  : forward predicted result
        f_g_12: forward ground truth
        b_21  : backward predicted result
        b_g_21: backward ground truth
    Return:
        Sum of forward L2-distance and backward L2-distance
    """
    f_12_loss = GVLoss(f_12, f_g_12)
    b_21_loss = GVLoss(b_21, b_g_21)
    loss = f_12_loss + b_21_loss
    return loss

# module unit test
if __name__ == '__main__':
    B = 10
    D = 1
    f_12   = torch.rand([B, D], requires_grad=True)
    b_21   = torch.rand([B, D], requires_grad=True)
    f_g_12 = torch.rand([B, D], requires_grad=True)
    b_g_21 = torch.rand([B, D], requires_grad=True)

    loss = GroupGVLoss(f_12, f_g_12, b_21, b_g_21)
    loss.backward()
    print('loss module test pass')

    

