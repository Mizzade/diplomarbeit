#!/bin/bash

dataset_name='webcam'
conv_feature_name='covariant_point_tilde'
feature_name='feature_point_tilde'
network_name='mexico_tilde_p24_Mexico_train_point_translation_iter_20'
stats_name='mexico_tilde_p24_Mexico_train_point'

point_number='1000'
matlab="/home/mizzade/Workspace/MATLAB/R2018b/bin/matlab"

#python patch_network_point_test.py --train_name $network_name --stats_name $stats_name --dataset_name $dataset_name --save_feature $conv_feature_name

$matlab -r "point_extractor('$dataset_name','$conv_feature_name','$feature_name',$point_number);";
#$matlab -r "point_extractor('$dataset_name','$conv_feature_name','$feature_name',$point_number);  exit(0);";
