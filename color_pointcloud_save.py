import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import copy
import cv2

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
    pcds = []
    rgb_list = []
    depth_list = []
    capture_idx = 2

    # Capture
    capture = k4a.get_capture()
    rgb = capture.color

    # select roit
    x, y, w, h = cv2.selectROI(rgb)

    # pointcloud roi mask
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    mask = mask.astype(np.int16)
    indicies = np.where(mask==1)
    mask = np.expand_dims(mask, axis=-1)
    
    cv2.destroyAllWindows()

    time.sleep(0.5)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])

    # Define the intrinsic parameters of the depth camera
    intrinsic_matrix = k4a._calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)

    print(intrinsic_matrix)
    
    width = rgb.shape[1]
    height = rgb.shape[0]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # Create Open3D camera intrinsic object
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width,
                                                          height=height,
                                                          fx=fx,
                                                          fy=fy,
                                                          cx=cx,
                                                          cy=cy)


    while True:
        idx += 1
        if idx == capture_idx + 1:
                break

        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        rgb = capture.color
        depth = capture.transformed_depth_point_cloud

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, :, :3].astype(np.uint8)

        depth = depth * mask
        rgb = rgb * mask
         
        rgb_list.append(rgb)
        depth_list.append(depth)

        time.sleep(1)
    
    for i in range(len(depth_list)):
        print('save pointclouds {0}'.format(i))
        rgb_image = rgb_list[i]
        depth_image = depth_list[i]

        # rgb image scaling 
        print('rgb image convert to pointclouds {0}'.format(i))
        rgb_image = rgb_image.astype(np.float32)
        rgb_image /= 255.
        rgb_image = np.reshape(rgb_image, [-1, 3])

        # depth image scaling
        print('depth image convert to pointclouds {0}'.format(i))
        depth_image = depth_image.astype(np.float32)
        depth_image /= 1000.
        depth_image = np.reshape(depth_image, [-1, 3])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(depth_image)
        # pcd.colors = o3d.utility.Vector3dVector(rgb_image)

        # down sample pointcloud
        # pcd.voxel_down_sample(0.01)

        # Statistical outlier removal
        print('remove outlier pointclouds {0}'.format(i))
        # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20,
        #                                  std_ratio=2.0)

        # Visualize the mesh
        o3d.visualization.draw_geometries([pcd])


        # Save point cloud
        o3d.io.write_point_cloud('./360degree_pointclouds/test_pointcloud_{0}.pcd'.format(i), pcd)