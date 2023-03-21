import open3d as o3d
import copy 
from modern_robotics import *
import cv2
from sfm.pose_utils import gen_poses

if __name__ == '__main__':

    
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    # Load point clouds
    pcds = []
    dir_name = '2023_03_20_17_35_50'
    gen_poses('./360degree_pointclouds/{0}/rgb/'.format(dir_name), 'exhaustive_matcher')

    # for i in range(27):
    #     rgb = cv2.imread('./360degree_pointclouds/{1}/rgb/test_rgb_{0}.png'.format(i, dir_name))
    #     pcd = o3d.io.read_point_cloud('./360degree_pointclouds/{1}/pcd/test_pointcloud_{0}.pcd'.format(i, dir_name))


    # # Visualize the mesh
    # o3d.visualization.draw_geometries(pcds)

    # cloud_base = pcds[0]

    # cloud1 = copy.deepcopy(cloud_base)


    # detectTransLoop = np.identity(4)
    # posWorldTrans = np.identity(4)

    # for cloud2 in pcds[1:]:

    #     posLocalTrans = registerLocalCloud(cloud1, cloud2)

    #     detectTransLoop = np.dot(posLocalTrans, detectTransLoop)

    #     posWorldTrans =  np.dot(posWorldTrans, posLocalTrans)

    #     cloud1 = copy.deepcopy(cloud2)
    #     cloud2.transform(posWorldTrans)
        
    #     cloud_base = cloud_base + cloud2
        
    #     # downsampling
    #     # cloud_base.voxel_down_sample(voxel_size)

    # o3d.visualization.draw_geometries([cloud_base])
    # cl, ind = cloud_base.remove_statistical_outlier(nb_neighbors=30, std_ratio=3.0)
    # cloud_base = cloud_base.select_by_index(ind)
    # o3d.visualization.draw_geometries([cloud_base])

    # # estimate normals
    # # cloud_base = cloud_base.voxel_down_sample(voxel_size)
    # # cloud_base.estimate_normals()
    
    # # cloud_base.orient_normals_to_align_with_direction()

    # # surface reconstruction
    # # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud_base, depth=9, n_threads=1)[0]
    # # mesh, des = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud_base, depth=15)

    

    # # cloud_base.estimate_normals()
    
    # print('Create 3d mesh use alpha shape')
    # # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud_base, 0.001)
    # # mesh.compute_vertex_normals()

    # cloud_base.compute_convex_hull()
    # cloud_base.estimate_normals()
    # cloud_base.orient_normals_consistent_tangent_plane(10)

    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud_base, depth=10, scale=1.1, linear_fit=False)[0]
    # bbox = cloud_base.get_axis_aligned_bounding_box()
    # mesh = mesh.crop(bbox)
    
    # mesh.compute_vertex_normals()
    # # mesh.paint_uniform_color([0.5, 0.5, 0.5])
    # mesh.remove_degenerate_triangles()
    # mesh.remove_duplicated_triangles()
    # mesh.remove_non_manifold_edges()
    # mesh.remove_duplicated_vertices()
    # o3d.visualization.draw_geometries([cloud_base, mesh], mesh_show_back_face=True)

    # # Visualize the mesh
    # print('Visualize the mesh')
    # # o3d.visualization.draw_geometries([mesh])

    # # Save point cloud & mesh
    # print('Visualize the mesh')
    # o3d.io.write_point_cloud('./360degree_pointclouds/{0}/mesh/merged_pointclouds.ply'.format(dir_name), cloud_base)
    # o3d.io.write_triangle_mesh('./360degree_pointclouds/{0}/mesh/3d_model.gltf'.format(dir_name), mesh)