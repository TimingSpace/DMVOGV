#!/bin/bash
cd /home/wangxiangwei/Program/SingleShotVO/src
traindata=kitti
testdata=kitti

python3 ssvo_test.py --imagelist ../dataset/$traindata/$traindata.train.image --motion ../dataset/$traindata/$traindata.train.pose --imagelist_test ../dataset/$testdata/$testdata.test.image --motion_test ../dataset/$testdata/$testdata.test.pose --ip http://128.237.141.242 --port 8528 --model_load ../saved_model//model_0913_002_095.pt --batch 100 --model 0913_002_091_09
