import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
from azure_kinect import PyAzureKinectCamera
import open3d as o3d
import time
import numpy as np
import copy
import cv2
import os
from datetime import datetime

if __name__ == "__main__":
    camera = PyAzureKinectCamera(resolution='1536')

    now = datetime.now()
    current_time = now.strftime('%Y_%m_%d_%H_%M_%S')

    save_dir = './360degree_pointclouds/{0}/'.format(current_time)
    save_raw_rgb_dir = save_dir + 'images/'
    save_rgb_dir = save_dir + 'rgb/'
    save_pcd_dir = save_dir + 'pcd/'
    save_mesh_dir = save_dir + 'mesh/'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_raw_rgb_dir, exist_ok=True)
    os.makedirs(save_rgb_dir, exist_ok=True)
    os.makedirs(save_pcd_dir, exist_ok=True)
    os.makedirs(save_mesh_dir, exist_ok=True)

    idx = 0
    pcds = []
    raw_rgb_list = []
    rgb_list = []
    depth_list = []
    capture_idx = 24

    # Capture
    camera.capture()
    rgb = camera.get_color()

    # select roit
    x, y, w, h = cv2.selectROI(rgb)

    # pointcloud roi mask
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    mask = mask.astype(np.int16)
    indicies = np.where(mask==1)
    # mask = np.expand_dims(mask, axis=-1)
    
    cv2.destroyAllWindows()

    time.sleep(0.5)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])

    # Define the intrinsic parameters of the depth camera
    intrinsic_matrix = camera.get_color_intrinsic_matrix()

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
        if idx == 20:
            break
        camera.capture()
        rgb = camera.get_color()
        depth = camera.get_transformed_depth()
        
        raw_rgb = rgb.copy()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, :, :3].astype(np.uint8)

        object_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

        roi_rgb = rgb.copy()[y:y+h, x:x+w]

        # object_mask[y:y+h, x:x+w] = 255

        # object_mask = (object_mask / 255.).astype(np.uint16)
        
        # depth *= object_mask.astype(np.uint16)
        # rgb *= np.expand_dims(object_mask.astype(np.uint8), axis=-1)
        
        cv2.imshow('test', rgb)

        key = cv2.waitKey(500)
        # if key == ord('q'):
        #     break
        # elif key == ord('d'):
        print('capture idx {0}'.format(idx))
        raw_rgb_list.append(raw_rgb)
        rgb_list.append(rgb)
        depth_list.append(depth)
        
        idx += 1

    
    for i in range(len(depth_list)):
        print('save pointclouds {0}'.format(i))
        rgb_image = rgb_list[i]
        save_rgb = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR)
        depth_image = depth_list[i]
        
        # rgb image scaling 
        rgb_image = rgb_image.astype('uint8')

        # convert rgb image to open3d depth map
        rgb_image = o3d.geometry.Image(rgb_image)

        # depth image scaling
        depth_image = depth_image.astype('uint16')
        
        # convert depth image to open3d depth map
        depth_image = o3d.geometry.Image(depth_image)
        
        # convert to rgbd image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image,
                                                                        depth_image,
                                                                        convert_rgb_to_intensity=False)

        test_rgbd_image = np.asarray(rgbd_image)

        print('rgbd shape', test_rgbd_image.shape)
    

        # rgbd image convert to pointcloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)

        # Save point cloud
        o3d.io.write_point_cloud(save_pcd_dir + 'test_pointcloud_{0}.pcd'.format(i), pcd)

        # Save raw rgb image
        cv2.imwrite(save_raw_rgb_dir + 'test_raw_rgb_{0}.png'.format(i), raw_rgb_list[i])

        # Save rgb image
        cv2.imwrite(save_rgb_dir + 'test_rgb_{0}.png'.format(i), save_rgb)