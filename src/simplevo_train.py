import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import transformation as tf
import data_loader
import loss_functions
import VONet
import evaluate
import options 
import matplotlib.pyplot as plt
torch.manual_seed(100) # random seed generate random number


def main():
    # parameters and flags
    args  = options.options

    ################## initial model###########################
    model = VONet.SPADVONet(coor_layer_flag = args['coor_layer_flag'], color_flag = args['color_flag'], vo_dimension= np.sum(args['vo_dimension']))
    model = model.float()
    print(model)

    if args['use_gpu_flag']:
        model     = model.cuda()
    if args['finetune_flag']:
        model.load_state_dict(torch.load(args['model_path']))

    ################## initial optimization ###########################   
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(200, 0,50).step)
    print(optimizer)

    ################### initial dataloader####################
    dataloader_train, dataloader_vis, dataloader_vid  = initial_dataloader(args)

    ################### initial logger   ####################
    training_loss_data = open('../checkpoint/saved_result/{}_training.loss'.format(args['training_tag']), 'a')
    testing_loss_data  = open('../checkpoint/saved_result/{}_testing.loss'.format(args['training_tag']), 'a')
    training_rpe_data  = open('../checkpoint/saved_result/{}_training.ate'.format(args['training_tag']), 'a')
    testing_rpe_data   = open('../checkpoint/saved_result/{}_testing.ate'.format(args['training_tag']), 'a')
     ################## training   #######################

    for epoch in range(101):
        print('epoch start')
        epoch_loss = 0
        result = []
        result = np.array(result)
        model.train()
        for i_batch, sample_batched in enumerate(dataloader_train):
            batch_loss,result = pad_update(model, sample_batched, use_gpu_flag=args['use_gpu_flag'], vo_dimension = args['vo_dimension'])
            epoch_loss += batch_loss
            print(epoch,'******',i_batch,'/',len(dataloader_train),'*******',batch_loss.item())
            batch_loss.backward()
            optimizer.step()

        lr_scheduler.step()
        if epoch % args['log_period'] ==0:
            model_saved_path = '../checkpoint/saved_model/model_{}_{}.pt'.format(args['training_tag'],str(epoch).zfill(3) )
            torch.save(model.state_dict(),model_saved_path )
            evaluate_model(model, dataloader_vis, training_loss_data, training_rpe_data, args)
            evaluate_model(model, dataloader_vid, testing_loss_data,  testing_rpe_data,  args)
        print('epoch end')


def evaluate_model(model, dataloader, log_loss_data, log_rpe_data, args):
    """
    To evaluate the trained model with given data
    Attributes:
        model: trained vo model
        dataloader: a dataloader for a dataset
        log_loss_data: for logging the loss
        log_rpe_data: for logging the relative pose error
        args: parameters in options
    Return:
        None
    Raise:
        None
    """
    with torch.no_grad():
        model.eval()
        forward_result = []
        ground_truth = []
        epoch_loss = 0
        for i_batch, sample_batched in enumerate(dataloader):
            model.zero_grad()
            batch_loss, result = pad_update(model,sample_batched,use_gpu_flag=args['use_gpu_flag'])
            #batch_loss.backward()
            batch_loss.detach_()
            log_loss_data.write(str(batch_loss.cpu().data.tolist())+'\n')
            log_loss_data.flush()
            epoch_loss +=  batch_loss
            temp_f  = weighted_mean_motion(result)
            gt_f_12 = sample_batched['motion_f_01'].numpy()
            forward_result = np.append(forward_result,temp_f)
            ground_truth = np.append(ground_truth,gt_f_12)

        forward_result = forward_result.reshape(-1,6)*dataloader.dataset.motion_stds
        ground_truth = ground_truth.reshape(-1,6)*dataloader.dataset.motion_stds

        forward_result_m = tf.eular2pose(forward_result)
        ground_truth_m          = tf.eular2pose(ground_truth)
        if args['rpe_flag']:
            rot_train,tra_train   = evaluate.evaluate(ground_truth_m,forward_result_m)
            log_rpe_data.write(str(np.mean(tra_train))+' '+ str(np.mean(rot_train))+'\n')
            log_rpe_data.flush()


class LambdaLR(object):
    """
    A class for reduce the learning rate
    """
    def __init__(self, n_epochs, offset, decay_start_epoch):
        """
        Constructor funtion:
        Attributes:
            n_epochs: the total epochs needed 
            offset: the learning rate left in the last
            decay_start_epoch: the epoch number when the decay begins 
        """
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        """
        Calculate the current learning rate
        Attributes:
            epoch: current epoch number
        Return:
            current learning rate
        Raise:
            None
        """
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def initial_dataloader(args):
    """
    Dataloader intialization functions
    Attribute:
        args: A dictionary contains the path of the data file, and mutiple flags
    Return:
        Three dataloader: 
            dataloader_train: dataloader of the training dataset, with shuffle
            dataloader_vis: dataloader of the training dataset, without shuffle
            dataloader_vid: dataloader of the testing dataset, without shuffle
    """
    # training data
    data_path = args['data_path']
    image_files_train = data_path + 'image.train' 
    image_files_test  = data_path + 'image.test' 
    pose_files_train  = data_path + 'pose.train' 
    pose_files_test   = data_path + 'pose.test' 
    camera_parameter = args['camera_parameter']
    image_size = (camera_parameter[1],camera_parameter[0])
    
    # training transform
    if args['color_flag']:
        transforms_ = [
                    transforms.Resize(image_size),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    else:
        transforms_ = [
                    transforms.Resize(image_size),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor()]
    kitti_dataset = data_loader.SepeDataset(path_to_poses_files=pose_files_train, 
                                            path_to_image_lists=image_files_train,
                                            transform_=transforms_, 
                                            camera_parameter=camera_parameter, 
                                            coor_layer_flag=args['coor_layer_flag'], 
                                            color_flag = args['color_flag'])

    dataloader_train = DataLoader(kitti_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=1, drop_last=True)
    if args['data_balance_flag']:
        print('data balance by prob')
        dataloader_train = DataLoader(kitti_dataset, batch_size=args['batch_size'], shuffle=False ,num_workers=1, drop_last=True, sampler=kitti_dataset.sampler)
    else:
        print('no data balance')    
    dataloader_vis = DataLoader(kitti_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=1, drop_last=True)
    # testing data
    # transform
    if args['color_flag']:
        transforms_ = [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    else:
        transforms_ = [
                    transforms.Resize(image_size),
                    transforms.ToTensor()]

    kitti_dataset_test = data_loader.SepeDataset(path_to_poses_files=pose_files_test,
                                                path_to_image_lists=image_files_test,
                                                transform_=transforms_,
                                                camera_parameter = camera_parameter,
                                                norm_flag=1,
                                                coor_layer_flag=args['coor_layer_flag'],
                                                color_flag=['color_flag'])

    dataloader_vid = DataLoader(kitti_dataset_test, batch_size=args['batch_size'], shuffle=False, num_workers=1, drop_last=True)

    return dataloader_train, dataloader_vis, dataloader_vid


def plot_path(poses,epoch,args):
    fig = plt.figure(figsize=(12,6.5))
    labels = ['Ground Truth','Estimation']
    i = 0
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for pose in poses:
        ax1.plot(pose[:,3],pose[:,11],label = labels[i])
        ax2.plot(pose[:,7],label = labels[i])
        i+=1
    ax1.legend(loc = 'upper left')
    ax1.set_xlabel('x/m')
    ax1.set_ylabel('z/m')
    ax2.set_xlabel('frame')
    ax2.set_xlabel('y/m')
    ax2.legend(loc = 'upper left')
    plt.savefig('../checkpoint/saved_result/testing_path_'+args.model_name+'_'+str(epoch).zfill(3)+'.png')


def weighted_mean_motion(predicted_result):
    predict_f_12 = predicted_result[0]
    result = np.zeros((predict_f_12.shape[0],6))
    if True:
        predict_b_21 = predicted_result[1]
        predict_f_12 = torch.stack((predict_f_12,-predict_b_21))
        predict_f_12 = predict_f_12.mean(0)
    temp_f = predict_f_12.cpu().data.numpy()
    result[:,[2,4]] = temp_f
    return result


def pad_update(model, sample_batched, use_gpu_flag=True, vo_dimension = [True, True, True, True, True, True]):
    """
    Training block
    Attributes:
        model: the model to be trained
        sample_batched: data for training the model
        use_gpu_flag: 
        vo_dimension: indicate which dimension of the motion is going to learn
    Return:
        [loss, vo_result]
    """
    model.zero_grad()
    input_batch_images_f_12  = sample_batched['image_f_01']
    input_batch_motions_f_12 = sample_batched['motion_f_01']
    input_batch_images_b_21  = sample_batched['image_b_10']
    input_batch_motions_b_21 = sample_batched['motion_b_10']

    if use_gpu_flag:
        input_batch_images_f_12 = input_batch_images_f_12.cuda()
        input_batch_motions_f_12 = input_batch_motions_f_12.cuda()
        input_batch_images_b_21 = input_batch_images_b_21.cuda()
        input_batch_motions_b_21 = input_batch_motions_b_21.cuda()

    predict_f_12 = model(input_batch_images_f_12)
    predict_b_21 = model(input_batch_images_b_21)
    result=[predict_f_12,predict_b_21]#,predict_f_13,att_f_13,predict_b_31,att_b_31,predict_f_23,att_f_23,predict_b_32,att_b_32]
    print(input_batch_motions_b_21.shape, input_batch_motions_b_21[:, vo_dimension].shape)
    batch_loss = loss_functions.GroupGVLoss(predict_f_12, input_batch_motions_f_12[:, vo_dimension], \
        predict_b_21,input_batch_motions_b_21[:, vo_dimension])

    return batch_loss,result


if __name__ == '__main__':
    main()





'''
def optimized_motion(predicted_result,kitti_dataset,ego_pre):
    predict_f_12 = predicted_result[0]
    predict_b_21 = predicted_result[2]
    temp_f = predict_f_12.cpu().data.numpy()
    temp_b = predict_b_21.cpu().data.numpy()
    temp_f = np.transpose(temp_f,(0,2,3,1))
    temp_f = temp_f*kitti_dataset_test.motion_stds+kitti_dataset_test.motion_means
    temp_f = np.transpose(temp_f,(0,3,1,2))

    temp_b = np.transpose(temp_b,(0,2,3,1))
    temp_b = temp_b*kitti_dataset_test.motion_stds+kitti_dataset_test.motion_means
    temp_b = np.transpose(temp_b,(0,3,1,2))

    quat_f = tf.sos2quats(temp_f[:,3:6,:,:])
    quat_b = tf.sos2quats(temp_b[:,3:6,:,:])
    quats = np.concatenate((quat_f,quat_b),axis=1)
    #print(quats.shape)

    trans_f = np.transpose(temp_f[:,0:3,:,:].reshape(temp_f.shape[0],3,temp_f.shape[2]*temp_f.shape[3]),(0,2,1))
    trans_b = np.transpose( temp_b[:,0:3,:,:].reshape(temp_b.shape[0],3,temp_b.shape[2]*temp_b.shape[3]),(0,2,1))
    trans = np.concatenate((trans_f,trans_b),axis=1)
    optimized_motion =[]
    for i_q in range(0,quats.shape[0]):
        ego = ego_pre.predict_patch(quats[i_q,:,:],trans[i_q,:,:])
        optimized_motion = np.append(forward_visual_opti,tf.SE2se(np.array(ego)))
    return optimized_motion
'''


