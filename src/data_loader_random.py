'''
* Copyright (c) 2019 Carnegie Mellon University, Author <xiangwew@andrew.cmu.edu> <basti@andrew.cmu.edu>
*
* Not licensed for commercial use. For research and evaluation only.
*
'''

from __future__ import print_function, division
import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import transformation as tf
from torch.utils.data.sampler import WeightedRandomSampler
#import transformation as tf


def coor_channel(camera_parameter):
    image = np.zeros((2,camera_parameter[1],camera_parameter[0]),dtype=float)
    for i_row in range(0,int(camera_parameter[1])):
        for i_col in range(0,int(camera_parameter[0])):
            image[0,i_row,i_col] = (i_row - camera_parameter[5])/camera_parameter[3]
            image[1,i_row,i_col] = (i_col - camera_parameter[4])/camera_parameter[2]
    return image
class SepeDataset(Dataset):
    """
    Dataset: the dataset can contain multiple sequences, for each sequence a list file
    pose file is necessary, the paths of list files and motion files should be in two txt
    files:
    """
    def __init__(self, path_to_poses_files, path_to_image_lists, transform_=None,camera_parameter=[640,180,640,640,320,90],norm_flag=0,coor_layer_flag=True):
        """
        Args:
            motions_file (string): Path to the pose file with camera pose.
            image_paths_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.poses_files = pd.read_csv(path_to_poses_files)
        self.image_lists   = pd.read_csv(path_to_image_lists)
        #assert len(self.poses_files) == len(self.image_lists)
        print('data file number',len(self.poses_files),len(self.image_lists))
        self.seq_num       = len(self.poses_files)
        
        self.poses         = []
        self.motions       = []
        self.image_paths   = []
        self.seq_lengths   = np.zeros((self.seq_num+1))
        all_ses = []
        self.coor_layer_flag = coor_layer_flag
        if self.coor_layer_flag:
            self.coor_channel = coor_channel(camera_parameter)
            self.coor_channel = torch.from_numpy(self.coor_channel).float()
        for i in range(0,self.seq_num):
            poses = np.loadtxt(self.poses_files['data'].loc[i])
            if poses.shape[1] == 7:# quaternion to SE3
                poses = tf.pos_quats2SEs(poses)

            #motions = tf.pose2motion(poses)
            #ses     = tf.SEs2ses(motions[0:-1,:])
            motions = tf.pose2motion(poses)
            ses     = tf.motion2eular(motions[0:-1,:])
            all_ses = np.append(all_ses,ses)
            self.motions.append(motions)
            self.poses.append(poses)
            self.image_paths.append(pd.read_csv(self.image_lists['data'].loc[i]))
            self.seq_lengths[i+1] =self.seq_lengths[i]+motions.shape[0]-1
        all_ses = all_ses.reshape(int(self.seq_lengths[-1]),6)
        weight = motion_weight(all_ses[:,4])
        samples_weight = torch.from_numpy(weight)
        samples_weigth = samples_weight.double()
        self.sampler = WeightedRandomSampler(samples_weight,len(samples_weight))
        self.transform = transforms.Compose(transform_)
        self.motion_means = np.mean(all_ses,0)
        self.motion_stds  = np.std(all_ses,0)
        mean_std_name = path_to_poses_files.split('.')
        print(mean_std_name)
        mean_std_name = '.'.join(mean_std_name[0:-2])+'.motion_mean_std.txt'
        print(mean_std_name)
        if norm_flag == 0:
            np.savetxt(mean_std_name,[self.motion_means,self.motion_stds])
        elif norm_flag == 1:
            std_mean = np.loadtxt(mean_std_name)
            self.motion_means = std_mean[0,:]
            self.motion_stds  = std_mean[1,:]
        print('motion mean and std',self.motion_means,self.motion_stds)

    def __len__(self):
        return int(self.seq_lengths[-1])
        #return 20

    def __getitem__(self, idx):
        #print(idx)
        for i in range(0,self.seq_num):
            if idx<self.seq_lengths[i+1]:
                #calculate sequence id and data id
                seq_length = self.seq_lengths[i+1] - self.seq_lengths[i]
                seq_id = i
                data_id = int(idx-self.seq_lengths[i])
                #load image
                lowest_delta = max(-5,-data_id)
                highest_delta = min(6,seq_length-data_id)
                choices = list(range(int(lowest_delta),int(highest_delta)))+[1,1,1,2,2]
                delta = np.random.choice(choices)
                img_name_0 = self.image_paths[seq_id]['data'].loc[data_id+0]
                img_name_1 = self.image_paths[seq_id]['data'].loc[data_id+delta]

                image_0 = Image.open(img_name_0).convert('RGB')
                image_1 = Image.open(img_name_1).convert('RGB')
                if(self.transform):
                    image_0 = self.transform(image_0)
                    image_1 = self.transform(image_1)
                image_f_01 = np.concatenate((image_0, image_1), axis=0)
                image_b_10 = np.concatenate((image_1, image_0), axis=0)
                if self.coor_layer_flag:
                    image_f_01 = np.concatenate((image_f_01,self.coor_channel),axis=0)
                    image_b_10 = np.concatenate((image_b_10,self.coor_channel),axis=0)

                #load motion
                pose_0_mat   =  np.matrix(np.eye(4))
                pose_1_mat   =  np.matrix(np.eye(4))
                pose_0_row   =  self.poses[seq_id][data_id+0,:]
                pose_1_row   =  self.poses[seq_id][data_id+delta,:]


                pose_0_mat[0:3,:]   =  np.matrix(pose_0_row.reshape(3,4))
                pose_1_mat[0:3,:]   =  np.matrix(pose_1_row.reshape(3,4))
                motion_f_01_mat = pose_0_mat.I*pose_1_mat
                motion_b_10_mat = pose_1_mat.I*pose_0_mat
                motion_01_row_6 = torch.Tensor(tf.mat2eular(motion_f_01_mat))
                motion_10_row_6 = torch.Tensor(tf.mat2eular(motion_b_10_mat))
                motion_means_tensor = torch.Tensor(self.motion_means)
                motion_stds_tensor = torch.Tensor(self.motion_stds)
                if False:
                    motion_01_row_6 = (motion_01_row_6-motion_means_tensor)/motion_stds_tensor
                    motion_10_row_6 = (motion_10_row_6-motion_means_tensor)/motion_stds_tensor
                elif True:
                    motion_01_row_6 = (motion_01_row_6)/motion_stds_tensor
                    motion_10_row_6 = (motion_10_row_6)/motion_stds_tensor

                sample = {'image_f_01': image_f_01,\
                    'image_b_10':image_b_10,\
                    'motion_f_01':motion_01_row_6,\
                    'motion_b_10':motion_10_row_6\
                    }

                return sample
            else:
                continue



def motion_weight(data):
    weight =[]
    hist,bins = np.histogram(data,bins=20)
    bins_min = np.min(bins)
    bins_width =bins[1]-bins[0]
    for d in data:
        n = int((d - bins_min-0.0000000000001)/bins_width)
        w = hist[n]
        weight.append(w)
    weight = np.array(weight)
    weight = 1/weight
    return weight

def main():
    motion_files_path = sys.argv[1]
    path_files_path = sys.argv[2]
    transforms_ = [
                transforms.Resize((376,1240)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]


    #kitti_dataset = KittiDataset(motions_file=motion_files_path,image_paths_file=path_files_path,transform=composed)
    kitti_dataset = SepeDataset(path_to_poses_files=motion_files_path,path_to_image_lists=path_files_path,transform_=transforms_)
    print(len(kitti_dataset))
    dataloader = DataLoader(kitti_dataset, batch_size=4,shuffle=False ,num_workers=1,drop_last=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image_f_01'],sample_batched['image_b_20'].size())
        print(i_batch, sample_batched['motion_f_01'],sample_batched['motion_b_20'])
if __name__== '__main__':
    main()