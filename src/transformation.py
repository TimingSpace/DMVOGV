'''
* Copyright (c) 2019 Carnegie Mellon University, Author <xiangwew@andrew.cmu.edu> <basti@andrew.cmu.edu>
*
* Not licensed for commercial use. For research and evaluation only.
*
'''
import numpy as np
from scipy.spatial.transform import Rotation as R

def line2mat(line_data):
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return np.matrix(mat)
def motion2pose(data):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = line2mat(data[i,:])
        pose = pose*data_mat
        pose_line = np.array(pose[0:3,:]).reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose

def pose2motion(data):
    data_size = data.shape[0]
    all_motion = np.zeros((data_size-1,12))
    for i in range(0,data_size-1):
        pose_curr = line2mat(data[i,:])
        pose_next = line2mat(data[i+1,:])
        motion = pose_curr.I*pose_next
        motion_line = np.array(motion[0:3,:]).reshape(1,12)
        all_motion[i,:] = motion_line
    return all_motion

def pose2motion2(data,ratio=1):
    data_size = data.shape[0]
    all_motion = np.zeros((data_size-1,12))
    for i in range(0,data_size-1):
        pose_curr = line2mat(data[i,:])
        pose_next = line2mat(data[i+1,:])
        motion = pose_curr.I*pose_next
        motion_eular = mat2eular(motion)
        _,_,_,r_x,r_y,r_z = motion_eular
        r_y*=ratio
        motion_R = eular2mat([r_x,r_y,r_z])
        motion_t = motion_R.I*motion[0:3,3]
        #z = np.sqrt(motion[0:3,3].T*motion[0:3,3])
        #motion_t =np.array([0,0,z])
        motion[0:3,3] = motion_t.reshape(3,1)
        motion_line = np.array(motion[0:3,:]).reshape(1,12)
        all_motion[i,:] = motion_line
    return all_motion

def motion2pose2(data,ratio=1):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        motion = line2mat(data[i,:])
        motion_eular = mat2eular(motion)
        _,_,_,r_x,r_y,r_z = motion_eular
        r_y*=ratio
        motion_R = eular2mat([r_x,r_y,r_z])
        motion_t = motion_R*motion[0:3,3]
        motion[0:3,3] = motion_t
        pose = pose*motion
        pose_line = np.array(pose[0:3,:]).reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose

def eular2pose(data):
    motion = eular2motion(data)
    pose = motion2pose(motion)
    return pose



def eular2pose2(data,ratio):
    motion = eular2motion(data)
    pose = motion2pose2(motion,ratio)
    return pose

def SE2se(SE_data):
    result = np.zeros((6))
    result[0:3] = np.array(SE_data[0:3,3].T)
    result[3:6] = SO2so(SE_data[0:3,0:3]).T
    return result
def SO2so(SO_data):
    return R.from_dcm(SO_data).as_rotvec()

def so2SO(so_data):
    return R.from_rotvec(so_data).as_dcm()

def se2SE(se_data):
    result_mat = np.matrix(np.eye(4))
    result_mat[0:3,0:3] = so2SO(se_data[3:6])
    result_mat[0:3,3]   = np.matrix(se_data[0:3]).T
    return result_mat
### can get wrong result
def se_mean(se_datas):
    all_SE = np.matrix(np.eye(4))
    for i in range(se_datas.shape[0]):
        se = se_datas[i,:]
        SE = se2SE(se)
        all_SE = all_SE*SE
    all_se = SE2se(all_SE)
    mean_se = all_se/se_datas.shape[0]
    return mean_se

def ses_mean(se_datas):
    se_datas = np.array(se_datas)
    se_datas = np.transpose(se_datas.reshape(se_datas.shape[0],se_datas.shape[1],se_datas.shape[2]*se_datas.shape[3]),(0,2,1))
    se_result = np.zeros((se_datas.shape[0],se_datas.shape[2]))
    for i in range(0,se_datas.shape[0]):
        mean_se = se_mean(se_datas[i,:,:])
        se_result[i,:] = mean_se
    return se_result

def ses2poses(data):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = se2SE(data[i,:])
        pose = pose*data_mat
        pose_line = np.array(pose[0:3,:]).reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose

def SEs2ses(motion_data):
    data_size = motion_data.shape[0]
    ses = np.zeros((data_size,6))
    for i in range(0,data_size):
        SE = np.matrix(np.eye(4))
        SE[0:3,:] = motion_data[i,:].reshape(3,4)
        ses[i,:] = SE2se(SE)
    return ses

def so2quat(so_data):
    so_data = np.array(so_data)
    theta = np.sqrt(np.sum(so_data*so_data))
    axis = so_data/theta
    quat=np.zeros(4)
    quat[0:3] = np.sin(theta/2)*axis
    quat[3] = np.cos(theta/2)
    return quat

def quat2so(quat_data):
    quat_data = np.array(quat_data)
    sin_half_theta = np.sqrt(np.sum(quat_data[0:3]*quat_data[0:3]))
    axis = quat_data[0:3]/sin_half_theta
    cos_half_theta = quat_data[3]
    theta = 2*np.arctan2(sin_half_theta,cos_half_theta)
    so = theta*axis
    return so

# input so_datas batch*channel*height*width
# return quat_datas batch*numner*channel
def sos2quats(so_datas,mean_std=[[1],[1]]):
    so_datas = np.array(so_datas)
    so_datas = so_datas.reshape(so_datas.shape[0],so_datas.shape[1],so_datas.shape[2]*so_datas.shape[3])
    so_datas = np.transpose(so_datas,(0,2,1))
    quat_datas = np.zeros((so_datas.shape[0],so_datas.shape[1],4))
    for i_b in range(0,so_datas.shape[0]):
        for i_p in range(0,so_datas.shape[1]):
            so_data = so_datas[i_b,i_p,:]
            quat_data = so2quat(so_data)
            quat_datas[i_b,i_p,:] = quat_data
    return quat_datas

def quat2SO(quat_data):
    return R.from_quat(quat_data).as_dcm()


def pos_quat2SE(quat_data):
    SO = R.from_quat(quat_data[3:7]).as_dcm()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3,:]).reshape(1,12)
    return SE


def pos_quats2SEs(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = np.zeros((data_len,12))
    for i_data in range(0,data_len):
        SE = pos_quat2SE(quat_datas[i_data,:])
        SEs[i_data,:] = SE
    return SEs


def mat2eular(data):
    t_x = data[0,3]
    t_y = data[1,3]
    t_z = data[2,3]
    r_y = -np.arcsin(data[2,0])
    cos_ry = np.cos(r_y)
    r_x = np.arctan2(data[2,1]/cos_ry,data[2,2]/cos_ry)
    r_z = np.arctan2(data[1,0]/cos_ry,data[0,0]/cos_ry) 
    return [t_x,t_y,t_z,r_x,r_y,r_z]

def eular2mat(data):
    r_x,r_y,r_z = data
    R_x = [[1,0,0],[0,np.cos(r_x),-np.sin(r_x)],[0,np.sin(r_x),np.cos(r_x)]]
    R_y = [[np.cos(r_y),0,np.sin(r_y)],[0,1,0],[-np.sin(r_y),0,np.cos(r_y)]]
    R_z = [[np.cos(r_z),-np.sin(r_z),0],[np.sin(r_z),np.cos(r_z),0],[0,0,1]]
    R = np.array(R_z)@np.array(R_y)@np.array(R_x)
    return np.matrix(R)


def motion2eular(data):
    r_y = -np.arcsin(data[:,8])
    cos_ry = np.cos(r_y)
    r_x = np.arctan2(data[:,9]/cos_ry,data[:,10]/cos_ry)
    r_z = np.arctan2(data[:,4]/cos_ry,data[:,0]/cos_ry) 
    t_x = data[:,3]
    t_y = data[:,7]
    t_z = data[:,11]
    eular = np.stack((t_x,t_y,t_z,r_x,r_y,r_z))
    return eular.transpose()
def eular2motion(data):
    R_x = np.zeros((len(data),3,3))
    R_x[:,0,0] = 1.0
    R_x[:,1,1] = np.cos(data[:,3])
    R_x[:,1,2] = -np.sin(data[:,3])
    R_x[:,2,1] = np.sin(data[:,3])
    R_x[:,2,2] = np.cos(data[:,3])
    R_y = np.zeros((len(data),3,3))
    R_y[:,1,1] = 1.0
    R_y[:,0,0] = np.cos(data[:,4])
    R_y[:,0,2] = np.sin(data[:,4])
    R_y[:,2,0] = -np.sin(data[:,4])
    R_y[:,2,2] = np.cos(data[:,4])
    R_z = np.zeros((len(data),3,3))
    R_z[:,2,2] = 1.0
    R_z[:,0,0] = np.cos(data[:,5])
    R_z[:,0,1] = -np.sin(data[:,5])
    R_z[:,1,0] = np.sin(data[:,5])
    R_z[:,1,1] = np.cos(data[:,5])
    R = R_z@R_y@R_x
    T = np.zeros((len(data),3,4))
    T[:,0:3,0:3] = R
    T[:,0:3,3]   = data[:,0:3]
    return T.reshape(len(data),12)


def append(data):
    data_0 = data[0].reshape(-1)
    for d in data[1:]:
        data_0 =np.append(data_0,d.reshape(-1))
    return data_0

def rescale(pose,scale):
    motion = pose2motion(pose)
    motion_trans = motion[:,3:12:4]
    speed = np.sqrt(np.sum(motion_trans*motion_trans,1))
    motion_trans = (scale*motion_trans.transpose()/(speed+0.0000001)).transpose()
    motion[:,3:12:4] = motion_trans
    pose = motion2pose(motion)
    return pose
