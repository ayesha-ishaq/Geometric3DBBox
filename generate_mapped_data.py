import pickle
import json
import os
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
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.eval.tracking.utils import category_to_tracking_name

# from utils.data_util import NuScenesClasses

class NumpyDecoder(json.JSONDecoder):
    """ Special json decoder for numpy types """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if '__numpy__' in obj:
            dtype = obj['__numpy__']
            data = obj['data']
            return np.array(data, dtype=dtype)
        return obj
    
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
        b = point[1] * 20 + 250 - AverageValue_y
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

        world_point = deepcopy(pc.points)
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
        world_point = world_point[:, mask]

        return points, coloring, world_point

def post_process_coords(corner_coords, imsize=(1600, 900)):
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

def generate_nusc_seq_data(nusc, scenes, sequences_by_name, out_dir, split):

    print('Generating detection and ground truth sequences...')
    result = []
    output_images = os.path.join(out_dir, split, 'images')
    Path(output_images).mkdir(parents=True, exist_ok=True)
    output_labels = os.path.join(out_dir, split, 'labelTxt')
    Path(output_labels).mkdir(parents=True, exist_ok=True)
    for scene_id, scene_name in enumerate(tqdm(scenes)):
            print('scene ID: ', scene_id)
            print('scene name:', scene_name)
            scene = sequences_by_name[scene_name]
            first_token = scene['first_sample_token']
            last_token = scene['last_sample_token']
            current_token = first_token

            scene_result = {}
            tracking_id_set = set()

            frame_id = 0
            while True:
                current_sample = nusc.get('sample', current_token)

                # Get ego pose data
                lidar_top_data = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
                pcl_path = os.path.join(nusc.dataroot, lidar_top_data['filename'])
                ego_pose_lidar = nusc.get('ego_pose', lidar_top_data['ego_pose_token'])
                ego_trans_lidar = np.array(ego_pose_lidar['translation'], dtype=np.float32)
                ego_timestamp = np.array(ego_pose_lidar['timestamp'], dtype=np.int)
                lidar_transform = nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])

                front_cam = nusc.get('sample_data', current_sample['data']['CAM_FRONT'])
                front_cam_filename = front_cam['filename']
                ego_pose_cam = nusc.get('ego_pose', front_cam['ego_pose_token'])
                camera_transform = nusc.get('calibrated_sensor', front_cam['calibrated_sensor_token'])
                camera_intrinsic = camera_transform['camera_intrinsic']
                
                projected_lidar_frame, color, lidar_points = map_pointcloud_to_image(nusc, current_sample['data']['LIDAR_TOP'],
                                        current_sample['data']['CAM_FRONT'])

                frame_ann_tokens = current_sample['anns']
                gt_trans = []
                gt_size = []
                gt_yaw = []
                gt_rot = []
                gt_class = []
                gt_track_token = []
                dets_2d = []
                frustrum_2d = []
                points_image = []
                frustrum_color = []
                average_x = []
                average_y = []

                gt_next_exist = []
                gt_next_trans = []
                gt_next_size = []
                gt_next_yaw = []

                scene_token = scene_name +'_'+ str(current_token)+'_'
                for ann_token in frame_ann_tokens:
                    ann = nusc.get('sample_annotation', ann_token)
                    if ann['category_name'].split('.')[0] == 'vehicle':
                        tracking_name = category_to_tracking_name(ann['category_name'])
                        if tracking_name is not None:
                            instance_token = ann['instance_token']
                            tracking_id_set.add(instance_token)

                            # get 2D bounding box for gt
                            box = Box(ann['translation'],
                                      ann['size'], 
                                      Quaternion(ann['rotation']))

                            box_3d = deepcopy(box)
                            # Move them to the ego-pose frame.
                            box.translate(-np.array(ego_pose_cam['translation']))
                            box.rotate(Quaternion(ego_pose_cam['rotation']).inverse)

                            # Move them to the calibrated sensor frame (camera).
                            box.translate(-np.array(camera_transform['translation']))
                            box.rotate(Quaternion(camera_transform['rotation']).inverse)

                            # Filter out the corners that are not in front of the calibrated sensor.
                            corners_3d = box.corners()
                            in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
                            corners_3d = corners_3d[:, in_front]

                            # Project 3d box to 2d.
                            corner_coords = view_points(corners_3d, np.array(camera_transform['camera_intrinsic'], dtype=np.float32),
                            True).T[:, :2].tolist()

                            # Keep only corners that fall within the image.
                            final_coords = post_process_coords(corner_coords)

                            # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
                            if final_coords is None:
                                continue
                            else:
                                min_x, min_y, max_x, max_y = final_coords
                                dets_2d.append([min_x, min_y, max_x, max_y])
                                mask = np.ones(projected_lidar_frame.shape[1], dtype=bool)
                                mask = np.logical_and(mask, projected_lidar_frame[0, :] > min_x)
                                mask = np.logical_and(mask, projected_lidar_frame[0, :] < max_x)
                                mask = np.logical_and(mask, projected_lidar_frame[1, :] > min_y)
                                mask = np.logical_and(mask, projected_lidar_frame[1, :] < max_y)
                                projected_points = projected_lidar_frame[:2, mask]
                                points = lidar_points[:, mask]
                                coloring = color[mask]
                                frustrum_2d.append(points)
                                points_image.append(projected_points)
                                frustrum_color.append(coloring)

                                # Move them to the ego-pose frame.
                                box_3d.translate(-np.array(ego_pose_lidar['translation']))
                                box_3d.rotate(Quaternion(ego_pose_lidar['rotation']).inverse)

                                # Move them to the calibrated sensor frame.
                                box_3d.translate(-np.array(lidar_transform['translation']))
                                box_3d.rotate(Quaternion(lidar_transform['rotation']).inverse)

                                corners_3d = box_3d.corners().T

                                averageValue_x = np.mean(points[0, :]) * 20
                                averageValue_y = np.mean(points[1, :]) * 20
                                average_x.append(averageValue_x)
                                average_y.append(averageValue_y)

                                left_point = np.linalg.inv(np.array(camera_intrinsic)).dot(np.array([min_x, 0, 1]).copy().T)[[0, 2]]
                                right_point = np.linalg.inv(np.array(camera_intrinsic)).dot(np.array([max_x, 0, 1]).copy().T)[[0, 2]]
                                
                                
                                img = np.ones((700, 700, 3), 'f4')
                                img = img * 175
                                img = draw_point_clouds(img, coloring, points.T, averageValue_x, averageValue_y)
                                img = draw_frustum_lr_line(img, left_point, right_point, averageValue_x, averageValue_y)

                                file_name = scene_token + str(len(average_x) - 1)
                                bev_img_path = os.path.join(output_images, file_name+'.png')
                                bev_label_path = os.path.join(output_labels, file_name+'.txt')
                                
                                flag = True
                                for i, point in enumerate(corners_3d):
                                    if i == 0 or i == 1 or i == 4 or i == 5:    
                                        point[0] = np.floor(point[0] * 20 + 250 - averageValue_x)
                                        point[1] = np.floor(point[1] * 20 + 250 - averageValue_y)

                                        if point[0] < 0 or point[1] < 0 or np.isnan(point[0]) or np.isnan(point[1]):
                                            flag = False
                                    
                                if flag:
                                    cv2.imwrite(bev_img_path, img)
                                    bev_gt = str(int(corners_3d[0][0])) + ", " + str(int(corners_3d[0][1])) + ", " + str(int(corners_3d[1][0])) + ", " + str(int(corners_3d[1][1])) + ", " + str(int(corners_3d[5][0])) + ", " + str(int(corners_3d[5][1])) + ", " + str(int(corners_3d[4][0])) + ", " + str(int(corners_3d[4][1])) + ", " + "vehicle, " + str(0)
                                    f = open(bev_label_path, "w")
                                    f.write(bev_gt)
                                    f.close()

                                gt_trans.append(ann['translation'])
                                gt_size.append(ann['size'])
                                gt_yaw.append([quaternion_yaw(Quaternion(ann['rotation']))])
                                gt_rot.append(ann['rotation'])
                                gt_track_token.append(instance_token)

                                next_ann_token = ann['next']
                                if next_ann_token == "":
                                    gt_next_exist.append(False)
                                    gt_next_trans.append([0.0, 0.0, 0.0])
                                    gt_next_size.append([0.0, 0.0, 0.0])
                                    gt_next_yaw.append([0.0])
                                else:
                                    gt_next_exist.append(True)
                                    next_ann = nusc.get('sample_annotation', next_ann_token)
                                    gt_next_trans.append(next_ann['translation'])
                                    gt_next_size.append(next_ann['size'])
                                    gt_next_yaw.append([quaternion_yaw(Quaternion(next_ann['rotation']))])

                frame_anns_dict = {
                    'translation': np.array(gt_trans, dtype=np.float32), # [M, 3]
                    'size': np.array(gt_size, dtype=np.float32), # [M, 3]
                    'yaw': np.array(gt_yaw, dtype=np.float32), # [M, 1]
                    'rotation': np.array(gt_rot, dtype=np.float32), # [M, 4]
                    'class': np.array(gt_class, dtype=np.int32), # [M]
                    'tracking_id': gt_track_token, # [M]
                    'next_exist': np.array(gt_next_exist, dtype=np.bool), # [M]
                    'next_translation': np.array(gt_next_trans, dtype=np.float32), # [M, 3]
                    'next_size': np.array(gt_next_size, dtype=np.float32), # [M, 3]
                    'next_yaw': np.array(gt_next_yaw, dtype=np.float32), # [M, 1]
                    'det_2d': np.array(dets_2d, dtype=np.float32),
                    'frustrum_points': frustrum_2d,
                    'image_points': points_image,
                    'frustrum_color': frustrum_color,
                    'average_x': average_x,
                    'average_y': average_y
                }

                frame_result = {'ground_truths': frame_anns_dict,
                                'num_gts': len(gt_trans), # int: M
                                'scene_id': scene_id,
                                'frame_id': frame_id,
                                'ego_translation': ego_trans_lidar,
                                'timestamp': ego_timestamp,
                                'sample_token': current_token,
                                'image_file': front_cam_filename,
                                'pcl_file': pcl_path 
                                }
                if scene_name in scene_result:
                    scene_result[scene_name].append(frame_result)
                else:
                    scene_result[scene_name] = [frame_result]


                if current_token == last_token:
                    break

                next_token = current_sample['next']
                current_token = next_token
                frame_id += 1

            assert len(scene_result[scene_name]) == scene['nbr_samples']
            
            ## Convert instance token to tacking id for the whole scene
            tracking_token_to_id = {}
            for i, tracking_id in enumerate(tracking_id_set):
                tracking_token_to_id.update({tracking_id: i})
            
            for frame_result in scene_result[scene_name]:
                for i, tracking_token in enumerate(frame_result['ground_truths']['tracking_id']):
                    tracking_id = tracking_token_to_id[tracking_token]
                    frame_result['ground_truths']['tracking_id'][i] = tracking_id
                frame_result['ground_truths']['tracking_id'] = \
                    np.array(frame_result['ground_truths']['tracking_id'], dtype=np.int32)

            result.append(scene_result)
    
    print('Done generating.')
    print('======')
    
    return result

def generate_nusc_data(version, dataset_dir, output_dir):

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
    # test_scenes = splits_to_scene_names[test_split]
    split = ['train', 'val']
    scenes = [train_scenes, val_scenes]
    # output_dirs = [output_dir / 'training', output_dir / 'validation']

    # Train and validation split
    for i, scene in enumerate(scenes):
        if i == 0:
            continue
        # output.mkdir(parents=True, exist_ok=True)
        print(split[i])
        # train_data[scene_id][frame_id] = 
        # {'detections': {'box', 'class', 'score'}, 'ground_truths': {'box', 'class', 'tracking_id'}}
        data = generate_nusc_seq_data(nusc, scene, sequences_by_name, output_dir, split[i])

        path_out = os.path.join(output_dir, 'groundtruth_val.json')
        
        data = json.dumps(data, cls=NumpyEncoder)
        with open(path_out, "w") as outfile:
            json.dump(data, outfile)
 

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Nuscenes data preprocessing')
    args.add_argument('--dataset_dir', default=None, type=str,
                      help='Directory where nuScenes dataset is stored')
    args.add_argument('--version', default="v1.0", type=str,
                      help='Version of nuScenes dataset')
    args.add_argument('--output_dir', default=None, type=str,
                      help='Directory where preprocessed json files will be stored')
    args = args.parse_args()

    generate_nusc_data(version=args.version,
                       dataset_dir=Path(args.dataset_dir),
                       output_dir=Path(args.output_dir))
