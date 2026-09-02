import config_utils
from contact_grasp_estimator import GraspEstimator
import os
import open3d as o3d
import logging
import numpy as np
from visualization_utils import visualize_grasps
from scipy.spatial.transform import Rotation

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class SO101GraspGenerator:

    def __init__(self):
        self.pcd_path = "/home/noor/so101-drake/assets/filtered.ply"

        self.T_cam_world = np.load("/home/noor/so101-drake/assets/cam_to_world.npy")

        self.ckpt_dir = "checkpoints/scene_test_2048_bs3_hor_sigma_001"
        self.forward_passes = 1
        self.global_config = config_utils.load_config(
            checkpoint_dir=self.ckpt_dir,
            batch_size=self.forward_passes,
            arg_configs=[]
        )

        self.scale = 0.08 / 0.045
        d_eff = 0.1034 / self.scale
        my_depth = 0.01
        offset = d_eff - my_depth
        self.offset_gripper_frame = np.eye(4)
        self.offset_gripper_frame[0, 3] = -0.015
        self.offset_gripper_frame[2, 3] = -offset

        self.aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-0.2, -0.1, 0.02), 
            max_bound=(0.0, 0.2, 0.1),
        )
        
    def start(self):
        logging.info("starting grasp generator...")
        # Build the model
        self.grasp_estimator = GraspEstimator(self.global_config)
        self.grasp_estimator.build_network()
    
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)
    
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
    
        # Load weights
        self.grasp_estimator.load_weights(self.sess, saver, self.ckpt_dir, mode='test')
        
        os.makedirs('results', exist_ok=True)

        # run inference once to warm-start
        self.inference()
        logging.info("started grasp generator")

    def inference(self) -> np.ndarray | None:
        filtered = o3d.io.read_point_cloud(self.pcd_path)
        obj_mask = self.aabb.get_point_indices_within_bounding_box(filtered.points)
        filtered.transform(self.T_cam_world)

        pc_segments = {
            1: np.asarray(filtered.select_by_index(obj_mask).points),
            # 2: np.asarray(filtered.select_by_index(obj_mask, invert=True).points)
        }

        pc_full = np.asarray(filtered.points)
        pc_colors = (np.asarray(filtered.colors) * 255).astype(np.uint8)

        pc_full *= self.scale
        pc_segments = {k: v * self.scale for k, v in pc_segments.items()}

        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(
            self.sess, pc_full, pc_segments=pc_segments, 
            local_regions=True, filter_grasps=True, 
            forward_passes=self.forward_passes
        )

        # logging.info(f"pred_grasps_cam: {pred_grasps_cam[1].shape}")
        # logging.info(f"scores: {scores[1]}")
        # logging.info(f"contact_pts: {contact_pts[1].shape}")

        if len(scores[1]) == 0:
            return None
        top_idx = np.argmax(scores[1])

        # visualize_grasps(
        #     pc_full, pred_grasps_cam, scores, 
        #     plot_opencv_cam=True, pc_colors=pc_colors
        # )
        # visualize_grasps(
        #     pc_full, 
        #     {1: pred_grasps_cam[1][top_idx:top_idx+1]}, 
        #     {1: scores[1][top_idx:top_idx+1]}, 
        #     plot_opencv_cam=True, pc_colors=pc_colors
        # )

        candidates = pred_grasps_cam[1]
        candidates[:, :3, 3] /= self.scale
        candidates = np.linalg.inv(self.T_cam_world) @ candidates
        candidates = candidates @ self.offset_gripper_frame
        top_candidate = candidates[top_idx]
        return top_candidate


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    grasp_generator = SO101GraspGenerator()
    grasp_generator.start()
    logging.info("starting inference loop...")
    for _ in range(5):
        candidate = grasp_generator.inference()
        logging.info(candidate)


if __name__ == "__main__":
    main()
