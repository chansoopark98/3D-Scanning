import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import copy
import cv2
from modern_robotics import *


if __name__ == "__main__":

    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True
        )
    )
    k4a.start()
    
    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    idx = 0
    pcd_list = []
    capture_idx = 4

    # Capture
    capture = k4a.get_capture()
    rgb = capture.color

    # ROI 선택
    x, y, w, h = cv2.selectROI(rgb)

    # 포인트 클라우드 ROI 마스크 생성
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    mask = mask.astype(np.int16)
    indicies = np.where(mask==1)
    mask = np.expand_dims(mask, axis=-1)
    
    cv2.destroyAllWindows()

    time.sleep(1)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, -1])

    while True:
        idx += 1
        # if idx == capture_idx + 1:
        if idx == capture_idx + 1:
            break
        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        rgb = capture.color
        raw_pcd = capture.transformed_depth_point_cloud
        
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, :, :3].astype(np.float32) / 255

        raw_pcd = raw_pcd * mask
        rgb = rgb * mask

        raw_pcd = np.reshape(raw_pcd, [-1, 3])
        rgb = np.reshape(rgb, [-1, 3])

        max_range_mask = np.where(np.logical_and(raw_pcd[:, 2]<550, raw_pcd[:, 2]>430))
        raw_pcd = raw_pcd[max_range_mask]
        rgb = rgb[max_range_mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_pcd)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Move point cloud to new origin coordinate
        # pcd.translate(translation_vector)

        # Compute mean distance of points from origin
        distances = np.sqrt(np.sum(np.square(np.asarray(pcd.points)), axis=1))
        mean_distance = np.mean(distances)
        pcd.scale(1 / mean_distance, center=pcd.get_center())
        
        center = pcd.get_center()
        new_origin = [0, 0, 0]
        translation_vector = np.subtract(new_origin, center)
        pcd.translate(translation_vector)

        pcd_list.append(pcd)
        
        time.sleep(6)

    for i in range(len(pcd_list)):
        print('save_pointcloud {0}'.format(i))
        # Save point cloud
        o3d.io.write_point_cloud('./test_pointclouds/test_pointcloud_{0}.pcd'.format(i), pcd_list[i])
