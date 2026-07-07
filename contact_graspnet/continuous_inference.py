import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2
import pyrealsense2 as rs
import open3d as o3d

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 5)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 5)
profile = pipeline.start(config)

device = profile.get_device()
depth_sensor = device.first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 5) # Default - 0, High Accuracy - 3, High Density - 4, Medium Density - 5
depth_scale = depth_sensor.get_depth_scale()

align_to = rs.stream.color
align = rs.align(align_to)

intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
cam_K = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1],
])
distCoeffs = np.asarray(intrinsics.coeffs)

transform = np.load('/home/noor/polyROIExtrinsics/cam_to_world.npy')

o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=848,
    height=480,
    fx=cam_K[0, 0],
    fy=cam_K[1, 1],
    cx=cam_K[0, 2],
    cy=cam_K[1, 2],
)
aabb = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=(-0.2, 0.0, 0.01), 
    max_bound=(-0.05, 0.2, 0.1),
)

scale = 0.08 / 0.045

def inference(global_config, checkpoint_dir, input_path, K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    
    os.makedirs('results', exist_ok=True)

    # Process example test scenes
    try:
        while True:
            # get frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth = depth_image * depth_scale
            rgb = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # compute segmap
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb), 
                o3d.geometry.Image(depth_image),
                depth_scale=1000.0,
                depth_trunc=100.0,
                convert_rgb_to_intensity=False,
            )
            pcd_full = o3d.geometry.PointCloud.create_from_rgbd_image(
                image=rgbd_image,
                intrinsic=o3d_intrinsics,
                extrinsic=transform,
                project_valid_depth_only=False
            )
            obj_mask = aabb.get_point_indices_within_bounding_box(pcd_full.points)
            segmap = np.zeros(848*480, dtype=bool)
            segmap[obj_mask] = True
            segmap = segmap.reshape((480, 848))

            # generate grasp poses
            pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
                depth, cam_K, segmap=segmap, rgb=rgb, 
                skip_border_objects=skip_border_objects, z_range=z_range
            )
            pc_full *= scale
            pc_segments = {k: v * scale for k, v in pc_segments.items()}
            pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
                sess, pc_full, pc_segments=pc_segments, 
                local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes
            )  

            # Visualize results          
            # show_image(rgb, segmap)
            # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

            # print(pred_grasps_cam)
            # print(scores)
            
            print(len(scores[True]))

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))

    inference(global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, z_range=eval(str(FLAGS.z_range)),
                K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id, 
                forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)

