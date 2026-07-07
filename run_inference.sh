#!/bin/bash

python contact_graspnet/continuous_inference.py \
    --local_regions \
    --filter_grasps
    # --np_path=/home/noor/lerobot/realsense/images/cgn_input_clutter.npz \
    # --arg_configs \
    #     TEST.first_thres:0.19 \
    #     TEST.second_thres:0.15 \
    #     TEST.filter_thres:0.0001
