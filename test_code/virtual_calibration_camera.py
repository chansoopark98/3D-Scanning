import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import copy
import cv2

def get_rotation_matrix(angle):
    theta = np.radians(angle) # rotation angle in degrees
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]])
    return R

def get_extrinsic_matrix(angle):
    """ set rotate """
    # Set up rotation matrix for turntable (assuming counterclockwise rotation around z-axis)
    theta = np.radians(angle) # rotation angle in degrees
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]])

    # Set up camera extrinsic matrix (assuming camera is at (0.43, 0, 0.05) relative to turntable center)
    t = np.array([0.043, 0, 0])
    # t = np.array([0, 0, 0])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = np.matmul(-R, t)

    return extrinsic
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
    depth_list = []
    rgb_list = []
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
    # mask = np.expand_dims(mask, axis=-1)
    
    cv2.destroyAllWindows()

    time.sleep(1)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, -0.5])

    # Define the intrinsic parameters of the depth camera
    intrinsic_matrix = k4a._calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)

    print(intrinsic_matrix)
    
    width = rgb.shape[1]
    height = rgb.shape[0]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    while True:
        if idx == capture_idx:
                break

        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        rgb = capture.color
        depth = capture.transformed_depth
        
        cv2.imshow('Pointcloud capture', depth.copy() / 1000)
        if cv2.waitKey(0) == ord('s'):
            idx += 1
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = rgb[:, :, :3].astype(np.float32) / 255

            depth = depth * mask
            rgb = rgb * np.expand_dims(mask, axis=-1)

            max_range_mask = np.where(np.logical_and(depth<550, depth>400), 1, 0)
            depth = depth * max_range_mask
            rgb = rgb * np.expand_dims(max_range_mask, axis=-1)
            print(depth.shape)
            rgb_list.append(rgb)
            depth_list.append(depth)

    

    
    # Create Open3D camera intrinsic object
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width,
                                                          height=height,
                                                          fx=fx,
                                                          fy=fy,
                                                          cx=cx,
                                                          cy=cy)


    output_pcd_list = [coord_frame]
    
    
    for i in range(len(depth_list)):
        print('save_pointcloud {0}'.format(i))
        rgb_image = rgb_list[i]
        depth_image = depth_list[i]


        # depth image scaling
        depth_image = depth_image.astype(np.float32) / 1000.
        o3d_depth = o3d.geometry.Image(depth_image)

        # Create a new point cloud from the depth image
        pcd_from_depth = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth,
            intrinsic=camera_intrinsics,
            depth_scale=0.001,
            depth_trunc=100.0,
            stride=1,
            project_valid_depth_only=True
        )
        
        pcd_from_depth.colors = o3d.utility.Vector3dVector(np.reshape(rgb_list[i], [-1, 3]))

        print('center', pcd_from_depth.get_center())
        # Move the point cloud to its center
        # pcd_from_depth.translate(-pcd_from_depth.get_center())
        if i == 0:
            base_center = pcd_from_depth.get_center()
        pcd_from_depth.translate(-base_center)

        extrinsic = get_extrinsic_matrix(angle=90* i)
        rotation = get_rotation_matrix(angle= 90 * i)
        # pcd_from_depth.transform(extrinsic)
        pcd_from_depth.rotate(rotation, center=pcd_from_depth.get_center())

        # Move the point cloud back to its original position
        # pcd_from_depth.translate(pcd_from_depth.get_center())


        theta = np.radians(180) # rotation angle in degrees
        R = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
        extrinsic[:3, :3] = R

        pcd_from_depth.transform(extrinsic)


        output_pcd_list.append(pcd_from_depth)
        # Visualize the mesh
        o3d.visualization.draw_geometries(output_pcd_list)

        # Save point cloud
        o3d.io.write_point_cloud('./4way_pointclouds/test_pointcloud_{0}.pcd'.format(i), pcd_from_depth)
