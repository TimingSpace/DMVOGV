import numpy as np
import sys
import transformation as tf
import matplotlib.pyplot as plt
for i in range(0,11):
    plt.figure(num=None, figsize=(10, 3.5), dpi=80, facecolor='w', edgecolor='k')
    file_name = 'dataset/kitti_gt/'+str(i).zfill(2)+'.txt'
    pose = np.loadtxt(file_name)
    motion_3 = tf.motion2eular(tf.pose2motion(pose))
    std_ = np.std(motion_3,0)
    print(std_)
    

    ax0 =plt.subplot(131)
    ax0.plot(pose[:,3],pose[:,11])
    ax0.set_xlabel('x/m')
    ax0.set_ylabel('y/m')
    min_x = np.min(pose[:,3:12:4],0)
    max_x = np.max(pose[:,3:12:4],0)
    mean_x= (min_x+max_x)/2
    diff_x = max_x -min_x
    max_diff = np.max(diff_x)
    print(min_x,max_x)
    ax0.set_xlim(mean_x[0]-max_diff/2-0.1*max_diff,mean_x[0]+max_diff/2+0.1*max_diff)
    ax0.set_ylim(mean_x[2]-max_diff/2-0.1*max_diff,mean_x[2]+max_diff/2+0.1*max_diff)

    plt.title('path '+ str(i).zfill(2))
    labels = 'x', 'y', 'z'
    sizes = std_[0:3]/np.sum(std_[0:3])

    ax1 = plt.subplot(132)
    ax1.pie(sizes,labels=labels, autopct='%1.1f%%',
            startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title('translation '+ str(i).zfill(2))
    labels = r'$\psi$', r'$\theta$', r'$\varphi$'
    sizes = std_[3:6]/np.sum(std_[3:6])
    ax2 = plt.subplot(133)
    ax2.pie(sizes,labels=labels, autopct='%1.1f%%',
            startangle=90)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title('rotation '+ str(i).zfill(2))
    plt.savefig('motion_variance_kitti'+str(i).zfill(2)+'.pdf')
    plt.show()



