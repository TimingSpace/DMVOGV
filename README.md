# DMVOGV
Deep Monocular Visual Odometry for Ground Vehicle

## Motivation
Most learning-based visual odometry aim to learn six degree-of-freedom motion despite the motion in training dataset is contrained. We simplify the learning targe by only learning the major motion of a ground vehicle. we contruct a very light model to learn the main motion of ground vehicle which can run in real-time on CPU.


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
* After the training finished, you will got validation path in `checkpoint/saved_result` and saved_model parameter in `checkpoint/saved_model`

###  4. test
* adjust parameter (batch size, testing data and model_to_load) in `test.sh` refer to `src/options.py`
* `sh test.sh`

### 5. Code 
* `src/simple_train.py` main script of the project, intialize model, optimization method, dataloader and train
* `src/VONet.py`  Realization of VO estimation model
* `src/loss_functions.py ` Loss function
* `src/data_loader.py` dataloader which only load consequent images, `src/data_loader_random.py`, dataloader which can load images with random distance to contruct an input image pair
* `src/options.py` options 
* `src/transformation.py` motion transformation: `eular<->mat<->so3<->quaternion`, `pose <->motion`, `line<->mat` and so on. 

