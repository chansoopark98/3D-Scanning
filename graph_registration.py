import open3d as o3d
import copy 
from modern_robotics import *

# test
voxel_size = 0.001
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1


# Load point clouds
def load_point_clouds(voxel_size=0.0, pcds_paths=None):
    pcds = []
    print('len demo_icp_pcds_paths:', len(pcds_paths))
    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    for path in pcds_paths:
        pcd = o3d.io.read_point_cloud(path)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds

def load_orginal_point_clouds(voxel_size=0.0, pcds_paths=None):
    pcds = []
    print('len demo_icp_pcds_paths:', len(pcds_paths))
    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    for path in pcds_paths:
        pcd = o3d.io.read_point_cloud(path)
        pcds.append(pcd)
    return pcds

def pairwise_registration(source, target, init_trans):

    source.estimate_normals()
    target.estimate_normals()

    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds) # 16
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            print('source id:', source_id)
            print('target id:', target_id)

            init_trans = np.identity(4)
            transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id],
                                                                        init_trans)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case

                odometry = np.dot((transformation_icp), odometry)

                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case -> connect any non-neighboring nodes
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    # Loop closure
    init_trans = np.identity(4)
    transformation_icp, information_icp = pairwise_registration(pcds[n_pcds-1], pcds[0],
                                                                init_trans)
    pose_graph.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(n_pcds-1,
                                                 0,
                                                 transformation_icp,
                                                 information_icp,
                                                 uncertain=False))
    return pose_graph



if __name__ == '__main__':
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    # Load point clouds
    pcds_down = []
    original_pcds = []
    dir_name = '2023_03_20_17_35_50'
    for i in range(27):
        pcd = o3d.io.read_point_cloud('./360degree_pointclouds/{1}/pcd/test_pointcloud_{0}.pcd'.format(i, dir_name))    

        original_pcds.append(copy.deepcopy(pcd))

        # Filtering
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.2)
        filteredpcd = pcd.select_by_index(ind)

        filteredpcd = filteredpcd.voxel_down_sample(voxel_size)

        pcds_down.append(filteredpcd)
    

    # Visualize the mesh
    o3d.visualization.draw_geometries(pcds_down)


    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        preference_loop_closure=2.0,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    print("Transform points and display")
    accumulated_pcd = o3d.geometry.PointCloud()
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        accumulated_pcd += pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    o3d.visualization.draw_geometries([accumulated_pcd])

    cl, ind = accumulated_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.2)
    accumulated_pcd = accumulated_pcd.select_by_index(ind)

    o3d.visualization.draw_geometries([accumulated_pcd])
    
    # Render
    vis = o3d.visualization.Visualizer()
    vis.create_window('3DReconstructed')

    for p in pcds_down:
        vis.add_geometry(p)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 1.5

    vis.run()
    vis.destroy_window()