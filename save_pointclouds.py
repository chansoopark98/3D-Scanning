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
    capture_idx = 24

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


    while cv2.waitKey(500) != ord('q'):

        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        rgb = capture.color
        depth = capture.transformed_depth

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, :, :3].astype(np.uint8)

        object_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

        roi_rgb = rgb.copy()[y:y+h, x:x+w]

        # 크로마키
        hsv = cv2.cvtColor(roi_rgb.copy(), cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (37, 109, 0), (70, 255, 255)) # 영상, 최솟값, 최댓값
        green_mask = cv2.bitwise_not(green_mask)

        object_mask[y:y+h, x:x+w] = green_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        object_mask = cv2.erode(object_mask, kernel, iterations=2)

        object_mask = (object_mask / 255.).astype(np.uint16)
        
        depth *= object_mask.astype(np.uint16)
        rgb *= np.expand_dims(object_mask.astype(np.uint8), axis=-1)

        cv2.imshow('test', rgb)

        rgb_list.append(rgb)
        depth_list.append(depth)

        # time.sleep(1)
    
    for i in range(len(depth_list)):
        print('save pointclouds {0}'.format(i))
        rgb_image = rgb_list[i]
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
        o3d.io.write_point_cloud('./360degree_pointclouds/test_pointcloud_{0}.pcd'.format(i), pcd)