import open3d as o3d
import numpy as np
import copy


# Camera parameters
camera_height = 0.05  # 5 cm
camera_radius = 0.42  # 42 cm

# Turntable parameters
turntable_radius = 0.4  # 80 cm / 2

# Capture parameters
num_captures = 24
capture_interval = 1.0  # 1 second

angular_velocity = 360.0 / (num_captures * capture_interval)  # degrees per second

angles = np.linspace(0, 360, num_captures, endpoint=False)
angles -= (angles[1] - angles[0]) / 2  # center the angles

points = []
for angle in angles:
    # Calculate the camera position
    x = camera_radius * np.sin(np.radians(angle))
    y = camera_height
    z = camera_radius * np.cos(np.radians(angle))

    # Add the camera position to the list of points
    points.append([x, y, z])


# Load point clouds
pointclouds = []
for i in range(24):
    pcd = o3d.io.read_point_cloud(f"./test_pointclouds/test_pointcloud_{i}.pcd")

    # Transform the point cloud to the global coordinate system
    pcd.transform(np.array(
        [[1, 0, 0, points[i][0]],
         [0, 1, 0, points[i][1]],
         [0, 0, 1, points[i][2]],
         [0, 0, 0, 1]]
    ))

    # Add the transformed point cloud to the list
    pointclouds.append(pcd)


with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:

    # Visualize the merged point cloud
    o3d.visualization.draw_geometries(pointclouds)

    # # Set the ICP parameters (you can tweak these as needed)
    # icp_max_distance = 0.05
    # icp_max_iterations = 200
    # icp_init_guess = o3d.geometry.TransformationMatrix.identity(4)

    # # Register the point clouds using ICP
    # global_registration = o3d.pipelines.registration.registration_icp(
    #     source=pointclouds[0],
    #     target=pointclouds[1],
    #     max_correspondence_distance=icp_max_distance,
    #     init=icp_init_guess,
    #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
    #         relative_fitness=1e-6,
    #         relative_rmse=1e-6,
    #         max_iteration=icp_max_iterations
    #     )
    # )

    # for i in range(1, num_captures - 1):
    #     # Transform the source point cloud using the previous transformation
    #     source_transformed = pointclouds[i].transform(global_registration.transformation)

    #     # Register the transformed source point cloud to the target point cloud
    #     icp_result = o3d.pipelines.registration.registration_icp(
    #         source=source
