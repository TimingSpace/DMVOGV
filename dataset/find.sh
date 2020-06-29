#!/bin/bash
pos=front
loc=/data/datasets/zhudelong/abs_data/CMUDeepVO/medium_forest/data/version_1
for seq in {0..11}
do
    seq_zf=$(printf %03d $seq)
    echo $seq_zf
    #find $loc/C$seq_zf/$pos\_center_rgb/ | sort > cmu_deepvo.$pos.C$seq_zf.image
    #cp $loc/C$seq_zf/$pos\_center_rgb.csv cmu_deepvo.$pos.C$seq_zf.pose
    python3 ../test_path.py cmu_deepvo.$pos.C$seq_zf.pose cmu_deepvo.$pos.C$seq_zf.SE
done
