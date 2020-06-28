import argparse
#configuration
def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--optimizer', dest='optimization_method', default='sgd', help='optimization method')
    parser.add_argument('--result', dest='result_path', default='result/00_predict_posang.txt', help='predict result path')
    parser.add_argument('--imagelist', dest='image_list_path', default='dataset/kitti_image_train.txt', help='image list path')
    parser.add_argument('--motion', dest='motion_path', default='dataset/kitti_pose_train.txt', help='motion path')
    parser.add_argument('--model', dest='model_name', default='test_att_from_0_00_simpleatt', help='model name')
    parser.add_argument('--model_load', dest='model_load', default='cmu_data_all', help='model name')
    parser.add_argument('--batch', dest='batch_size',type=int, default=40, help='batch size')
    parser.add_argument('--motion_test', dest='motion_path_test', default='dataset/kitti_pose_test.txt', help='test motion path')
    parser.add_argument('--imagelist_test', dest='image_list_path_test', default='dataset/kitti_image_test.txt', help='test image list path')
    parser.add_argument('--port', dest='visdom_port', default='8202', help='visdom port')
    parser.add_argument('--ip', dest='visdom_ip', default='http://128.237.139.184', help='visdom port')
    parser.add_argument('--mean_std_path', dest='mean_std_path', default='my_utils/mean_std.txt', help='visdom port')
    args = parser.parse_args()
    return args
