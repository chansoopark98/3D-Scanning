import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A, FPS
import open3d as o3d
import time
import numpy as np
import copy
import cv2

class PyAzureKinectCamera(object):
    def __init__(self) -> None:
        config = self.get_camera_config(resolution='1080')
        self.k4a = PyK4A(config=config)
        self.k4a.start()

        self.capture_buffer = None
    
    def get_camera_config(self, resolution: str = '720'):
        if resolution == '720':
            color_resolution = pyk4a.ColorResolution.RES_720P
        elif resolution == '1080':
            color_resolution = pyk4a.ColorResolution.RES_1080P
        elif resolution == '1440':
            color_resolution = pyk4a.ColorResolution.RES_1440P
        elif resolution == '2160':
            color_resolution = pyk4a.ColorResolution.RES_2160P
        else:
            raise ValueError('설정한 카메라 해상도를 찾을 수 없습니다.\
                 현재 입력한 해상도 {0}'.format(resolution))

        config = pyk4a.Config(
            color_resolution=color_resolution,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
            camera_fps=FPS.FPS_5
        )
        return config

    def get_color_intrinsic_matrix(self) -> np.ndarray:
        return self.k4a._calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
    
    def get_depth_intrinsic_matrix(self) -> np.ndarray:
        return self.k4a._calibration.get_camera_matrix(pyk4a.CalibrationType.DEPTH)
        
    def capture(self) -> None:
        self.capture_buffer = self.k4a.get_capture()

    def get_color(self) -> np.ndarray:
        return self.capture_buffer.color
    
    def get_depth(self) -> np.ndarray:
        return self.capture_buffer.depth
    
    def get_pcd(self) -> np.ndarray:
        return self.capture_buffer.depth_point_cloud

    def get_transformed_color(self) -> np.ndarray:
        return self.capture_buffer.transformed_color
        
    def get_transformed_depth(self) -> np.ndarray:
        return self.capture_buffer.transformed_depth
    
    def get_transformed_pcd(self) -> np.ndarray:
        return self.capture_buffer.transformed_depth_point_cloud
        
if __name__ == "__main__":

    idx = 0
    pcds = []
    rgb_list = []
    depth_list = []
    capture_idx = 8

    # Capture
    
    camera = PyAzureKinectCamera()
    camera.capture()
    
    rgb = camera.get_transformed_color()

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
    
    intrinsic_matrix = camera.get_depth_intrinsic_matrix()
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
        camera.capture()

        rgb = camera.get_transformed_color()
        depth = camera.get_depth()
        
        depth = np.where(depth>=700, 0, depth)

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
        object_mask = cv2.erode(object_mask, kernel)

        object_mask = (object_mask / 255.).astype(np.uint16)
        
        depth *= object_mask.astype(np.uint16)
        rgb *= np.expand_dims(object_mask.astype(np.uint8), axis=-1)

        rgb_list.append(rgb)
        depth_list.append(depth)

        time.sleep(2.998)
    
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

        # o3d.visualization.draw_geometries([pcd])
        
        # Statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20,
                                         std_ratio=3.0)

        # o3d.visualization.draw_geometries([pcd])

        # Save point cloud
        o3d.io.write_point_cloud('./360degree_pointclouds/test_pointcloud_{0}.pcd'.format(i), pcd)