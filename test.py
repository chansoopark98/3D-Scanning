import open3d as o3d
import numpy as np
import copy

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


coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
# Load point clouds
pcd_list = []
for i in range(24):
    pcd = o3d.io.read_point_cloud(f"./test_pointclouds/test_pointcloud_{i}.pcd")
    pcd_list.append(pcd)


test_pcds = copy.deepcopy(pcd_list)
test_pcds.append(coord_frame)
# Visualize point cloud
o3d.visualization.draw_geometries(test_pcds)


# Get center of point cloud to wrap around
# pcd_ref = pcd_list[0]
# center = pcd_ref.get_center()


# Set up geometry
points = np.zeros((0,3))
colors = np.zeros((0,3))
rotated_pcds = []
for i in range(24):
    pcd = pcd_list[i]

    # # Set center of new point cloud to be the same as the center of the reference point cloud
    # pcd.translate(-center)

    # Rotate point cloud by angle around the z-axis
    # angle은 라디안값
    angle_velocity = 2 * np.pi / 24
    angle = angle_velocity * i

    # Define rotation matrix
    R = np.array([[np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]])

    # Apply rotation to point cloud
    pcd.rotate(R)

    r = 1 # radius of turntable
    theta = np.radians(angle) + np.pi / 2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = 0
    pcd.translate((x, y, z))

    rotated_pcds.append(pcd)

# Visualize the merged point cloud
o3d.visualization.draw_geometries(rotated_pcds)

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    print('포인트 클라우드 정합 시작')
    
    # 변수 초기화
    detectTransLoop = np.identity(4)
    posWorldTrans = np.identity(4)

    # cloud_base (정합시 기준이 되는 포인트 클라우드)
    cloud_base = rotated_pcds[0]
    target = copy.deepcopy(cloud_base)
    source_pcds = rotated_pcds[1:]

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