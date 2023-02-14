import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import cv2
from typing import Optional, Tuple
from open3d_icp import full_registration

voxel_size = 0.05
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

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
    depth_list = []
    pcds = []
    icp_idx = 0

    """
        R[0], R[1], R[2], T[0],
        R[3], R[4], R[5], T[1],
        R[6], R[7], R[8], T[2]
    """
    while True:
        idx += 1
        if idx > 6:
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
        
        depth_list.append(output_depth)

        cv2.imshow('test', align_depth)
        if cv2.waitKey(1000) == ord('q'): # q를 누르면 종료   
            break
    
    cv2.destroyAllWindows()

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
    

    for depth_idx in range(len(depth_list)):
        print('Convert depth to pcs. Try index = {0}'.format(depth_idx))
        # Load the aligned depth image
        depth_image = depth_list[depth_idx]

        # Calculate the 3D coordinates of each pixel
        raw_pcd = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.float32)
        for v in range(depth_image.shape[0]):
            for u in range(depth_image.shape[1]):
                depth = depth_image[v, u]
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth
                raw_pcd[v, u] = np.array([x, y, z], dtype=np.float32)
        
        raw_pcd = np.reshape(raw_pcd, [-1, 3])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_pcd)
        # pcd.voxel_down_sample(voxel_size)
        pcds.append(pcd)

    # Align the point clouds using ICP
    calc_pcds = []
    source = pcds[0]
    source.estimate_normals()
    calc_pcds.append(source)
    
    for target in pcds[1:]:
        icp_idx += 1
        print('Calculate ICP. Try index = {0}'.format(icp_idx))
        target.estimate_normals()
        # ICP
        result = o3d.pipelines.registration.registration_icp(
            source, target, 0.02, np.identity(4),
             o3d.pipelines.registration.TransformationEstimationPointToPoint())

        target.transform(result.transformation)
        calc_pcds.append(target)
        # print('result => {0}'.format(result))

    # Merge the point clouds
    merged_pcd = o3d.geometry.PointCloud()
    for calc_pcd in calc_pcds:
        merged_pcd += calc_pcd

    # Downsample the point cloud
    downpcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Create the mesh
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(downpcd, depth=8)

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])