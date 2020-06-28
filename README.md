# DMVOGV
Deep Monocular Visual Odometry for Ground Vehicle

## Motivation
Most learning-based visual odometry aim to learn six degree-of-freedom motion despite the motion in training dataset is contrained. We simplify the learning targe by only learning the major motion of a ground vehicle.


## How to run this code
###  1. install dependency

###  2. download training dataset
* download kitti odometry dataset [http://www.cvlibs.net/datasets/kitti/eval_odometry.php](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
* put image list file in `dataset/kitti/`

###  3. train
* adjust parameter in `train.sh` refer to `src/options.py`
* `sh train.sh`

###  4. test
* `sh test.sh`

## Demo
