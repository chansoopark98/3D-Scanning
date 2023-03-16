import open3d as o3d
import numpy as np

# Load point clouds from multiple frames
pcd_list = []
for i in range(num_frames):
    pcd = o3d.io.read_point_cloud(f"frame_{i}.ply")
    pcd_list.append(pcd)

# Combine point clouds into a single point cloud
combined_pcd = o3d.geometry.PointCloud()
for pcd in pcd_list:
    combined_pcd += pcd

# Remove points below a certain height
height_threshold = 0.05  # adjust as needed
bounding_box = combined_pcd.get_axis_aligned_bounding_box()
min_height = bounding_box.min_bound[2]
remove_indices = []
for i, point in enumerate(combined_pcd.points):
    if point[2] - min_height < height_threshold:
        remove_indices.append(i)
combined_pcd = combined_pcd.select_by_index(remove_indices, invert=True)

# Fit a plane to the remaining points
plane_model, inliers = combined_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
inlier_pcd = combined_pcd.select_by_index(inliers)
plane = plane_model.parameters

# Calculate the normal vector of the plane
normal = plane[:3] / np.linalg.norm(plane[:3])

# Create a flat point cloud base
flat_base_points = []
for x, y in np.ndindex((100, 100)):
    point = np.array([x * 0.01, y * 0.01, -plane[3] / plane[2]])
    flat_base_points.append(point)
flat_base = o3d.geometry.PointCloud()
flat_base.points = o3d.utility.Vector3dVector(flat_base_points)

# Merge the flat base with the original point cloud
merged_pcd = combined_pcd + flat_base

# Visualize the result
o3d.visualization.draw_geometries([merged_pcd])
