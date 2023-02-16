import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import copy
import cv2
from modern_robotics import *

def turntable_translate(angle):
    # 턴테이블 지름(미터)
    diameter = 0.8

    # 턴테이블 중심과 카메라 렌즈까지의 거리
    distance = 0.425

    # Calculate the horizontal and vertical displacements of the turntable
    horizontal_displacement = diameter / 2 * np.cos(angle)
    vertical_displacement = diameter / 2 * np.sin(angle)

    # Calculate the x, y, and z coordinates of the center of the turntable in the coordinate system of the point cloud
    x = distance * np.sin(np.arccos(horizontal_displacement / distance))
    y = distance * np.sin(np.arccos(vertical_displacement / distance))
    z = distance * np.cos(np.arccos(horizontal_displacement / distance)) * np.cos(np.arccos(vertical_displacement / distance))

    # Create the translation vector
    translation_vector = np.array([x, y, z])


def calc_turntable_matrix(time_index, total_time):
    angle_velocity = 2 * np.pi / total_time
    angle = angle_velocity * time_index
    angle = 1.57 * time_index
    # Create a rotation matrix that describes the rotation of the turntable
    rotation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]])
    
    # # Set the translation component of the transformation matrix to the position of the turntable
    translation_vector = np.array([0, 0, 0])  # replace with the position of the turntable

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation_vector

    # # Combine the rotation and translation matrices to create the transformation matrix
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)

    print('matrix', transformation_matrix)
    return transformation_matrix

def cal_angle(pl_norm, R_dir):
    angle_in_radians = \
        np.arccos(
            np.abs(pl_norm.x*R_dir[0]+ pl_norm.y*R_dir[1] + pl_norm.z*R_dir[2])
            )

    return angle_in_radians


def register_pcds(target, source):
    icp_max_distance = 0.02
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    source_temp.voxel_down_sample(0.004)
    target_temp.voxel_down_sample(0.004)

    source_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    target_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    # source_temp.estimate_normals()
    # target_temp.estimate_normals()

    current_transformation = np.identity(4)
    # use Point-to-plane ICP registeration to obtain initial pose guess

    result_icp_p2l = o3d.pipelines.registration.registration_icp(
                source_temp, target_temp, icp_max_distance, current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())

    print("----------------")
    print("initial guess from Point-to-plane ICP registeration")
    print(result_icp_p2l)
    print(result_icp_p2l.transformation)

    p2l_init_trans_guess = result_icp_p2l.transformation
    print("----------------")

    result_icp = o3d.pipelines.registration.registration_icp(source_temp, target_temp, 0.01,
                p2l_init_trans_guess, o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result_icp.transformation


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
    capture_idx = 2

    # Capture
    capture = k4a.get_capture()
    rgb = capture.color

    # Select ROI
    x, y, w, h = cv2.selectROI(rgb)

    # Create mask
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    mask = mask.astype(np.int16)
    mask = np.expand_dims(mask, axis=-1)
    
    cv2.destroyAllWindows()

    time.sleep(1)

    start_time = time.time()
    while True:
        idx += 1
        # if idx == capture_idx + 1:
        if idx == capture_idx + 1:
            break
        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        rgb = capture.color
        raw_pcd = capture.transformed_depth_point_cloud

        current_time = time.time()

        
        
        raw_pcd = raw_pcd * mask
        # plt.imshow(raw_pcd/ 1000)
        # plt.show()

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

        trans_matrix = calc_turntable_matrix(float(idx-1), capture_idx)
        pcd.transform(trans_matrix)

        # pcd = pcd.voxel_down_sample(voxel_size=0.05)
        
        time.sleep(1)
        pcds.append(pcd)

    # Visualize the merged point cloud
    # Create coordinate system geometry
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=15, origin=[0, 0, 0])
        
    o3d.visualization.draw_geometries(pcds)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        print('포인트 클라우드 정합 시작')
        
        # 변수 초기화
        detectTransLoop = np.identity(4)
        posWorldTrans = np.identity(4)

        # cloud_base (정합시 기준이 되는 포인트 클라우드)
        cloud_base = pcds[0]
        target = copy.deepcopy(cloud_base)
        source_pcds = pcds[1:]

        for source in source_pcds:
            posLocalTrans = register_pcds(target=target, source=source)

            detectTransLoop = np.dot(posLocalTrans, detectTransLoop)
            posWorldTrans =  np.dot(posWorldTrans, posLocalTrans)

            # update latest cloud
            target = copy.deepcopy(source)
            source.transform(posWorldTrans)
            cloud_base = cloud_base + source

            # downsampling
            cloud_base.voxel_down_sample(0.001)

        # Visualize the merged point cloud
        o3d.visualization.draw_geometries([cloud_base])