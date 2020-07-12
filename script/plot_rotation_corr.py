import numpy  as np
import transformation as tf
import sys
import matplotlib.pyplot as plt
import evaluate_vo as evo



def get_norm(data):
    return  np.sqrt(np.sum(data*data,1))


# @input motions nx6 [t_0,t_1,t_2,r_0,r_1,r_2]
def focusing(motions,t=[0,1],r=[3,5]):
    motions[:,t] =0
    rotation_angle =  get_norm(motions[:,3:6]) 
    motions[:,r] = 0
    #motions[:,4]   = rotation_angle
    return motions


def distribution(data):
   #hist,bins = np.histogram(data,bins=20)
   #print(hist,bins)
   #plt.plot(bins[1:],hist)
   plt.hist(data,bins=20)
   #plt.show()

def plot_path(poses):
    for pose in poses:
        plt.plot(pose[:,3],pose[:,11])
    #plt.show()
def test():
    data_test = np.ones((10,3))
    print(get_norm(data_test))

def angle_pattern(motion):
    trans = motion[:,0:3]
    trans_norm = get_norm(trans)
    theta_t = np.arcsin(trans[:,0]/(trans_norm+0.0001))
    flag = (motion[:,2]>0.1)&(np.abs(motion[:,4])>0.05)
    plt.plot(theta_t[flag],label='translation yaw')
    plt.plot(motion[flag,4],label='rotation yaw')
    plt.plot(trans_norm[flag],label='t z')
    l = (theta_t[flag]/motion[flag,4] - 0.5)*trans_norm[flag]
    plt.plot(l)
    plt.legend()
    #plt.show()
    return theta_t, motion[:,4]


def main():
    all_t_theta = []
    all_r_theta = []
    for i in range(0,11):
        plt.figure(num=None, figsize=(10, 3.5), dpi=80, facecolor='w', edgecolor='k')
        pose_name = 'dataset/kitti_gt/'+str(i).zfill(2)+'.txt'
        pose    = np.loadtxt(pose_name)
        motion_mat = tf.pose2motion(pose)
        motion  = tf.SEs2ses(tf.pose2motion(pose))
        motion_4 = tf.motion2eular(tf.pose2motion(pose))
        plt.plot(motion_4[:,4],label = 'y-axis rotation (rad)')
        plt.plot(motion_4[:,0],'--',label = 'x-axis translation (m)')
        plt.legend()
        plt.xlabel('frame')
        plt.ylabel('motion')
        plt.title('Motion Correlation' + str(i).zfill(2))
        plt.savefig('motion_correlation__kitti'+str(i).zfill(2)+'.pdf')
        plt.show()
if __name__ == '__main__':
    main()
