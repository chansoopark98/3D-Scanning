import open3d as o3d
import numpy as np
import copy

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.05, origin=[0, 0, 0])

# Calculate the rotation angle for each point cloud (assuming 24 seconds for full rotation)
theta_0 = 0
theta_90 = 90 * np.pi / 180
theta_180 = 180 * np.pi / 180
theta_270 = 270 * np.pi / 180

def get_rotation_matrix_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    return R

# Apply transformations to register pcd90, pcd180, and pcd270 to pcd0
T_0to90 = get_rotation_matrix_y(theta_90 - theta_0)
T_0to180 = get_rotation_matrix_y(theta_180 - theta_0)
T_0to270 = get_rotation_matrix_y(theta_270 - theta_0)

rotation_matrix = [T_0to90, T_0to180, T_0to270]

# Load point clouds
pointclouds = []
for i in range(4):
    pcd = o3d.io.read_point_cloud(f"./4way_pointclouds/test_pointcloud_{i}.pcd")
    pointclouds.append(pcd)

# Define the camera-to-object distance and base distance
distance = 0.42  # distance in meters
base_distance = 0.42  # distance when point cloud at 0 degrees was captured

# Calculate the depth adjustment factor
depth_adjustment = distance / base_distance


with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:

    output_pcds = [coord_frame]
    output_pcds.append(pointclouds[0])
    for pcd_idx in range(len(pointclouds[1:])):
        new_pcd = pointclouds[pcd_idx+1]
        new_pcd.transform(rotation_matrix[pcd_idx])

        if pcd_idx == 1:
                # Create a transformation matrix that flips the z-axis
            flip_transform = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])

            # Apply the transformation matrix to the point cloud
            # new_pcd.transform(flip_transform)
        output_pcds.append(new_pcd)

        # Visualize the merged point cloud
        o3d.visualization.draw_geometries([new_pcd, coord_frame])
        o3d.visualization.draw_geometries(output_pcds)
    o3d.visualization.draw_geometries(output_pcds)
