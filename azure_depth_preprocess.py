import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import cv2
from typing import Optional, Tuple

def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img

if __name__ == "__main__":

    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=False
        )
    )
    k4a.start()

    # k4a.calibration.get_camera_matrix
    
    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    idx = 0
    pcds = []
    icp_idx = 0

    """
        R[0], R[1], R[2], T[0],
        R[3], R[4], R[5], T[1],
        R[6], R[7], R[8], T[2]
    """
    while True:
        idx += 1
        if idx == 36:
            break
        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()

        align_depth = capture.depth
        align_depth = np.where(align_depth>600, 0, align_depth)
        align_depth = align_depth.astype(np.float32)
        align_depth /= 1000
        align_depth *= 255
        align_depth = align_depth.astype(np.uint8)
        align_depth = np.where(align_depth>95, align_depth, 0)
        
        output_depth = align_depth.copy()
        output_depth = output_depth.astype(np.float32)
        output_depth /= 255

        cv2.imshow('test', align_depth)
        if cv2.waitKey(0) == ord('q'): # q를 누르면 종료   
            break

    # Define the intrinsic parameters of the depth camera
    intrinsic_matrix = k4a._calibration.get_camera_matrix(pyk4a.CalibrationType.DEPTH)
        
    """
        camera intrinsic matrix =>
            [[503.90097046   0.         323.00683594]
            [  0.         503.88079834 340.80682373]
            [  0.           0.           1.        ]]
    """

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # Load the aligned depth image
    depth_image = output_depth

    # Calculate the 3D coordinates of each pixel
    pcd = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.float32)
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            depth = depth_image[v, u]
            x = (u - cx) * depth / fx
            y = (v - cy) * depth / fy
            z = depth
            pcd[v, u] = np.array([x, y, z], dtype=np.float32)
    
    
    pcd *= 1000
    pcd = pcd.astype(np.int16)
    print(pcd.shape)
    raw_pcd = np.reshape(pcd, [-1, 3])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_pcd)
    pcd.voxel_down_sample(0.02)

    o3d.visualization.draw_geometries([pcd])
