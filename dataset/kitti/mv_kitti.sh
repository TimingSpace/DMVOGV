#!/bin/bash

for seq in 00 01 02 03 04 05 06 07 08 09 10
do
    mv $seq.txt kitti_pose_$seq.txt
done
    
