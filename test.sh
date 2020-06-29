#!/bin/bash
cd $PWD/src # path to current source 
traindata=kitti
testdata=kitti
tag=0628_002
vis_port=8501

python3 simplevo_test.py --imagelist ../dataset/$traindata/$traindata.image.train --motion ../dataset/$traindata/$traindata.pose.train --imagelist_test ../dataset/$testdata/$testdata.image.test --motion_test ../dataset/$testdata/$testdata.pose.test --ip http://127.0.0.1 --port $vis_port --model $tag --batch 2 --model_load test_save_model_2.pt #--fine_tune 0 --pad 1

