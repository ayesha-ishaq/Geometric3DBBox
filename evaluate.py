import pickle
import json
import os
from lang_sam import LangSAM
import torch
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from collections import OrderedDict
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from shapely.geometry import MultiPoint, box
from nuscenes.utils.geometry_utils import view_points, transform_matrix
# from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.common.config import config_factory

from lidar_detection_evaluation.nuscenes_eval_core import NuScenesEval

import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmrotate.models import build_detector
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset

def initialize_bev_detector(cfg_file, ckpoint):
    # Choose to use a config and initialize the detector
    config = cfg_file
    # Setup a checkpoint file to load
    checkpoint = ckpoint
    # Set the device to be used for evaluation
    device='cuda:0'
    # Load the config
    config = mmcv.Config.fromfile(config)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None
    # Initialize the detector
    model = build_detector(config.model)
    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)
    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']
    # We need to set the model's cfg for inference
    model.cfg = config
    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()
    return model

@ROTATED_DATASETS.register_module()
class LidarDataset(DOTADataset):
    """SAR ship dataset for detection."""
    CLASSES = ('vehicle',)

def depth_to_color(depth, min_depth=1, max_depth=50, colormap="viridis"):
  """
  This function takes a depth value (between min_depth and max_depth) and maps it to a color
  using the specified colormap.

  Args:
    depth: The depth value (between min_depth and max_depth).
    min_depth: The minimum depth value (default is 1).
    max_depth: The maximum depth value (default is 100).
    colormap: The name of the colormap to use (default is "viridis").

  Returns:
    A list containing the RGB values of the mapped color.
  """

  # Normalize the depth value between 0 and 1
  normalized_depth = (depth - min_depth) / (max_depth - min_depth)

  # Import necessary libraries
  import matplotlib.cm as cm
  from matplotlib.colors import rgb2hex

  # Get the color from the chosen colormap
  color = cm.get_cmap(colormap)(normalized_depth)

  return color[:3]

def draw_bbox_3d_to_2d_gt(img, corner, AverageValue_x, AverageValue_y):
    
    gt_box_3d = corner.copy()
    for point in gt_box_3d:
        point[0] = point[0] * 20 + 250 - AverageValue_x
        point[1] = point[1] * 20 + 250 - AverageValue_y

    cv2.line(img, (int(gt_box_3d[0][0]), int(gt_box_3d[0][1])), (int(gt_box_3d[1][0]), int(gt_box_3d[1][1])), (255, 0, 0), 1, 4)
    cv2.line(img, (int(gt_box_3d[1][0]), int(gt_box_3d[1][1])), (int(gt_box_3d[5][0]), int(gt_box_3d[5][1])), (255, 0, 0), 1, 4)
    cv2.line(img, (int(gt_box_3d[5][0]), int(gt_box_3d[5][1])), (int(gt_box_3d[4][0]), int(gt_box_3d[4][1])), (255, 0, 0), 1, 4)
    cv2.line(img, (int(gt_box_3d[4][0]), int(gt_box_3d[4][1])), (int(gt_box_3d[0][0]), int(gt_box_3d[0][1])), (255, 0, 0), 1, 4)
    
    return img

def draw_point_clouds(img, colors, KeyPoint_for_draw, AverageValue_x, AverageValue_y):

    for i, point in enumerate(KeyPoint_for_draw):
        a = point[0] * 20 + 250 - AverageValue_x
        b = point[2] * 20 + 250 - AverageValue_y
        color = np.array(depth_to_color(colors[i]))*255
        cv2.circle(img, (int(a), int(b)), 1, color, 2)
    
    return img

def draw_frustum_lr_line(img, left_point, right_point, AverageValue_x, AverageValue_y):
    
    left_point_draw = np.array([0., 0.])
    left_point_draw[0] = (left_point[0] * 20000 + 250 - AverageValue_x)
    left_point_draw[1] = (left_point[1] * 20000 + 250 - AverageValue_y)

    right_point_draw = np.array([0., 0.])
    right_point_draw[0] = (right_point[0] * 20000 + 250 - AverageValue_x)
    right_point_draw[1] = (right_point[1] * 20000 + 250 - AverageValue_y)

    initial_point_draw = np.array([0., 0.])
    initial_point_draw[0] = 250 - AverageValue_x
    initial_point_draw[1] = 250 - AverageValue_y
    
    cv2.line(img, tuple(initial_point_draw.astype(np.int32)), tuple(left_point_draw.astype(np.int32)),
             (255, 0, 0), 1, 4)
    cv2.line(img, tuple(initial_point_draw.astype(np.int32)), tuple(right_point_draw.astype(np.int32)),
             (255, 0, 0), 1, 4)

    return img

def calculate_height(top_1, top_2, bot_1, bot_2, keypoint):

    # calculate the [vertical] height in frustum at key vertex (input variable [keypoint])

    # because top and bottom plane of frustum crosses (0, 0, 0), we assume the plane equation: Ax + By + 1 * z = 0
    # |x1 y1| |A|     |-1|         |A|     |x1 y1| -1    |-1|
    # |     | | |  =  |  |     =>  | |  =  |     |    *  |  |
    # |x2 y2| |B|     |-1|         |B|     |x2 y2|       |-1|

    mat_1 = np.array([[top_1[0], top_1[1]], [top_2[0], top_2[1]]])
    mat_2 = np.array([[bot_1[0], bot_1[1]], [bot_2[0], bot_2[1]]])
    mat_3 = np.array([-1., -1.]).T

    top_plane_info = np.linalg.inv(mat_1).dot(mat_3)
    bot_plane_info = np.linalg.inv(mat_2).dot(mat_3)

    top_y = -1 * (keypoint[0] * top_plane_info[0] + keypoint[1] * 1) / top_plane_info[1]
    bot_y = -1 * (keypoint[0] * bot_plane_info[0] + keypoint[1] * 1) / bot_plane_info[1]

    return top_y, bot_y

def map_pointcloud_to_image(nusc, pointsensor_token: str,
                            camera_token: str,
                            min_dist: float = 1.0):
        """
        Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
        plane.
        :param pointsensor_token: Lidar/radar sample_data token.
        :param camera_token: Camera sample_data token.
        :param min_dist: Distance from the camera below which points are discarded.
        :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """

        cam = nusc.get('sample_data', camera_token)
        pointsensor = nusc.get('sample_data', pointsensor_token)
        pcl_path = os.path.join(nusc.dataroot, pointsensor['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        im = Image.open(os.path.join(nusc.dataroot, cam['filename']))

        # world_point = deepcopy(pc.points)
        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]
        coloring = depths
      
        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        coloring = coloring[mask]
        camera_point = pc.points[:3,mask] #world_point[:, mask]

        return points, coloring, camera_point

def get_box_lidar(box, poserecord, cs_record):
    
    # First step: transform from camera to ego frame
    box.rotate(Quaternion(cs_record['rotation']))
    box.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    box.rotate(Quaternion(poserecord['rotation']))
    box.translate(np.array(poserecord['translation']))

    return box

def generate_nusc_seq_data(nusc, cfg, checkpoint, scenes, sequences_by_name):
    
    print('Generating detection and ground truth sequences...')
    submission = {
    "meta": {
        "use_camera": True,
        "use_lidar":   True,
        "use_radar":  False,
        "use_map":     False,
        "use_external": False
        },
    "results": {}
    }
    model = LangSAM()
    orientedrcnn_model = initialize_bev_detector(cfg, checkpoint)
    text_prompt = ["car","truck","bus","trailer","construction_vehicle","motorcycle","bicycle"]
    cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for scene_id, scene_name in enumerate(tqdm(scenes)):
        if scene_id < 151:
            scene = sequences_by_name[scene_name]
            first_token = scene['first_sample_token']
            last_token = scene['last_sample_token']
            current_token = first_token   
            frame_id = 0
            while True:
                submission['results'][current_token] = []
                current_sample = nusc.get('sample', current_token)
    
                # Get ego pose data
                lidar_top_data = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
                ego_pose = nusc.get('ego_pose', lidar_top_data['ego_pose_token'])
                
                for cam in cameras:
                    camera = nusc.get('sample_data', current_sample['data'][cam])
                    poserecord = nusc.get('ego_pose', camera['ego_pose_token'])
                    camera_transform = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
                    camera_intrinsic = camera_transform['camera_intrinsic']
                    img_path = os.path.join(nusc.dataroot, camera['filename'])
                    ego_pose_cam = nusc.get('ego_pose', camera['ego_pose_token'])
                    image = Image.open(img_path).convert("RGB")
                    projected_lidar_frame, color, lidar_points = map_pointcloud_to_image(nusc, current_sample['data']['LIDAR_TOP'],
                        current_sample['data'][cam])
                    for prompt in text_prompt:
                        masks, boxes, phrases, logits = model.predict(image, prompt)
                        for i, mask in enumerate(masks):
                            if np.array(image).shape[:2] != mask.shape:
                                raise ValueError("Image and mask dimensions must be equal.")

                            # Extract mask indices
                            mask_indices = np.array(np.where(mask))

                            # Define a dtype with x and y integers    
                            arr1 = np.empty(projected_lidar_frame.shape[1], dtype=[('x', int), ('y', int)])
                            arr2 = np.empty(mask_indices.shape[1], dtype=[('x', int), ('y', int)])

                            # Add the data to the structured array
                            arr1['x'] = projected_lidar_frame[1, :].astype(int)
                            arr1['y'] = projected_lidar_frame[0, :].astype(int)

                            arr2['x'] = mask_indices[0, :]
                            arr2['y'] = mask_indices[1, :]
                            # Finding intersection
                            mask_points = np.in1d(arr1, arr2)

                            filtered_points = projected_lidar_frame[:2, mask_points]
                            filtered_lidar = lidar_points[:, mask_points]
                            filtered_colors = color[mask_points]

                            averageValue_x = np.mean(filtered_lidar[0, :]) * 20
                            averageValue_y = np.mean(filtered_lidar[2, :]) * 20
                        
                            x0, y0, x1, y1 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                            left_point = np.linalg.inv(np.array(camera_intrinsic)).dot(np.array([x0, 0, 1]).copy().T)[[0, 2]]
                            right_point = np.linalg.inv(np.array(camera_intrinsic)).dot(np.array([x1, 0, 1]).copy().T)[[0, 2]]
                            mat_1 = np.array([[left_point[0], right_point[0]], [left_point[1], right_point[1]]])

                            img = np.ones((700, 700, 3), 'f4')
                            img = img * 175
                            img = draw_point_clouds(img, filtered_colors, filtered_lidar.T, averageValue_x, averageValue_y)
                            img = draw_frustum_lr_line(img, left_point, right_point, averageValue_x, averageValue_y)
                            bev_result = inference_detector(orientedrcnn_model, img) # highest score detection
                            for result in bev_result:
                                for each in result:
                                    if each[0] == 0 or each[1] == 0:
                                        continue
                                    else:
                                        center_x = (each[0] + averageValue_x - 250)/20
                                        center_y = (each[1] + averageValue_y - 250)/20
                                        width = each[2]/20
                                        length = each[3]/20
                                        angle = -each[4]
                                        xc, yc, w, h, ag, conf = each[:6]
                                        break
                            wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
                            hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
                            p_x = ((xc - wx - hx)+ averageValue_x - 250)/20
                            p_y = ((yc - wy - hy) + averageValue_y - 250)/20

                            top_1 = np.linalg.inv(np.array(camera_intrinsic)).dot(np.array([x0, y0, 1]).T)
                            top_2 = np.linalg.inv(np.array(camera_intrinsic)).dot(np.array([x1, y0, 1]).T)
                            bot_1 = np.linalg.inv(np.array(camera_intrinsic)).dot(np.array([x0, y1, 1]).T)
                            bot_2 = np.linalg.inv(np.array(camera_intrinsic)).dot(np.array([x1, y1, 1]).T)

                            y_min, y_max = calculate_height(top_1, top_2, bot_1, bot_2, [p_x, p_y])
                            height = abs(y_max - y_min)
                            center_z = (y_max + y_min)/2

                            center = np.array([center_x, center_z, center_y])
                            size = np.array([height, width, length])

                            if np.any(np.isnan(center)) or np.any(np.isnan(size)) or np.isnan(angle):
                                continue

                            box = Box(center,
                                size, 
                                Quaternion(axis=[0, -1, 0], angle=-angle))
                            box = get_box_lidar(box, poserecord, camera_transform)
                            sample_result = {
                                            "sample_token": current_token,
                                            "translation": box.center.tolist(),
                                            "size": box.wlh.tolist(),
                                            "rotation": box.orientation.elements.tolist(),
                                            "velocity": [0.0, 0.0],
                                            "detection_name": prompt,
                                            "detection_score": float(logits[i]),
                                            "attribute_name": ''
                            }
                            submission['results'][current_token].append(sample_result)
                if current_token == last_token:
                    break
                next_token = current_sample['next']
                current_token = next_token
                frame_id += 1
            
    print('Done generating.')
    print('======')
    
    return submission

def generate_nusc_data(version, dataset_dir, cfg, checkpoint, output_dir, eval_only=False):

    dataset_dir = dataset_dir / version

    version_fullname = version
    if version == "v1.0":
        version_fullname += '-trainval'
    nusc = NuScenes(version=version_fullname, dataroot=dataset_dir, verbose=True)
    # nusc_test = NuScenes(version='v1.0-test', dataroot=dataset_dir, verbose=True)
    sequences_by_name = {scene["name"]: scene for scene in nusc.scene}
    # sequences_by_name.update({scene["name"]: scene for scene in nusc_test.scene})
    splits_to_scene_names = create_splits_scenes()

    train_split = 'train' if version == "v1.0" else 'mini_train'
    val_split = 'val' if version == "v1.0" else 'mini_val'
    test_split = 'test'
    train_scenes = splits_to_scene_names[train_split]
    val_scenes = splits_to_scene_names[val_split]
    test_scenes = splits_to_scene_names[test_split]
    result_file = output_dir / 'detection_result.json'
    pred_dir = output_dir / 'preds'
    gt_dir = output_dir / 'gt'

    if not eval_only:
        data = generate_nusc_seq_data(nusc, cfg, checkpoint, val_scenes, sequences_by_name)
        
        with open(result_file, "w") as f:
            json.dump(data, f)

        print("Data written to detection_result.json successfully!")
    
    with open(result_file) as f:
        result = json.load(f)
    
    for frame in result['results']:
        frame_path = frame + ".txt"
        path = pred_dir / frame_path
        f = open(path, "a")
        for detection in result['results'][frame]:
            x, y, z = detection['translation']
            w, l, h = detection['size']
            class_name = 'vehicle'
            rotation = Quaternion(np.array(detection['rotation'])).radians
            score = detection['detection_score']
            content = f"{class_name} {x} {y} {z} {l} {w} {h} {rotation} {score} \n"
            f.write(content)

    NuScenesEval(str(pred_dir), str(gt_dir), "class x y z l w h r score" , str(output_dir),
                 max_range=0.0,
                 min_score=0.0)
    # cfg = config_factory('detection_cvpr_2019')
    # verbose = True
    # nusc_eval = NuScenesEval(
    #     nusc,
    #     config=cfg,
    #     result_path=result_file,
    #     eval_set='val',
    #     output_dir=output_dir,
    #     verbose=verbose
    # )

    # metrics = nusc_eval.main()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Nuscenes data preprocessing')
    args.add_argument('--dataset_dir', default=None, type=str,
                      help='Directory where nuScenes dataset is stored')
    args.add_argument('--version', default="v1.0", type=str,
                      help='Version of nuScenes dataset')
    args.add_argument('--checkpoint', default=None, type=str,
                      help='Path to checkpoint for orientedRCNN')
    args.add_argument('--config', default=None, type=str,
                      help='Path to config for orientedRCNN')
    args.add_argument('--output_dir', default=None, type=str,
                      help='Directory where inference result is stored')
    args.add_argument('--eval_only', action="store_true", default=False,
                      help='Whether to perform evalution only and no inference')
    args = args.parse_args()

    generate_nusc_data(version=args.version,
                       dataset_dir=Path(args.dataset_dir),
                       cfg=args.config,
                       checkpoint=args.checkpoint,
                       output_dir=Path(args.output_dir),
                       eval_only=args.eval_only)