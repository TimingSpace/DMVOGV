# DMVOGV
Deep Monocular Visual Odometry for Ground Vehicle

## Motivation
Most learning-based visual odometry aim to learn six degree-of-freedom motion despite the motion in training dataset is contrained. We simplify the learning targe by only learning the major motion of a ground vehicle.


## How to run this code
###  1. install dependency (Supposed you are with a new Ubuntu with nothing installed)
* `sudo apt-get update `
* ` sudo apt-get install python3 python3-pip`
* `python3 -m pip install -r requirements`
* check whether you have a GPU on your computer 
	* if yes (you have a GPU with 2G or more graphic memory) you have to install nvidia driver, cuda by yourself;
	* according to [pytorch offical](https://pytorch.org/), install corresponding pytorch and torchvision



###  2. download training dataset
* download kitti odometry dataset [http://www.cvlibs.net/datasets/kitti/eval_odometry.php](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
* put image list file in `dataset/kitti/`

###  3. train
* adjust parameter (batch size, training data, evaluation data and so on) in `train.sh` refer to `src/options.py`
* `sh train.sh`

###  4. test
* `sh test.sh`

## Demo
