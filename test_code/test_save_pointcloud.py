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
    # mask = np.expand_dims(mask, axis=-1)
    
    cv2.destroyAllWindows()

    time.sleep(1)

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
        depth = capture.transformed_depth
    
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, :, :3].astype(np.float32)

        depth = depth * mask
        rgb = rgb * np.expand_dims(mask, axis=-1)

        max_range_mask = np.where(np.logical_and(depth<710, depth>500), 1, 0)
        depth = depth * max_range_mask
        rgb = rgb * np.expand_dims(max_range_mask, axis=-1)
        print(depth.shape)
        rgb_list.append(rgb)
        depth_list.append(depth)

        time.sleep(0.5)
    
    for i in range(len(depth_list)):
        print('save pointclouds {0}'.format(i))
        rgb_image = rgb_list[i]
        depth_image = depth_list[i]

        # rgbd image scaling
        rgb_image = rgb_image
        depth_image = np.expand_dims(depth_image.astype(np.float32) / 1000., axis=-1)

        
        print('rgb shape', rgb_image.shape)
        print('depth_image shape', depth_image.shape)
        # rgbd_image = np.concatenate([rgb_image, depth_image], axis=-1)
        
        # rgb_image = rgb_image.astype(np.uint8)
        # depth_image = depth_image.astype(np.uint8)
        print(np.mean(rgb_image))
        print(np.mean(depth_image))

        rgb_image = o3d.geometry.Image(rgb_image.astype(np.uint16))
        depth_image = o3d.geometry.Image(depth_image)

        # Create a new point cloud from the depth image
        pcd_from_depth = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            intrinsic=camera_intrinsics,
            depth_scale=0.001,
            depth_trunc=100.0,
            stride=1,
            project_valid_depth_only=False
        )

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image,
                                                                        depth_image,
                                                                        depth_scale=0.001,
                                                                        depth_trunc=100.0,
                                                                        convert_rgb_to_intensity=False)
        
        color_array = np.asarray(rgbd_image / 255)
        print(color_array.shape)
        plt.imshow(color_array)
        plt.show()

        color_array = np.reshape(color_array, [-1, 3])
        print(color_array.shape)
        # pcd_from_depth.colors = 

        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_from_depth.points))
        pcd.colors = o3d.utility.Vector3dVector(color_array)

        # Visualize the mesh
        o3d.visualization.draw_geometries([pcd])

        # Save point cloud
        o3d.io.write_point_cloud('./360degree_pointclouds/test_pointcloud_{0}.pcd'.format(i), pcd)