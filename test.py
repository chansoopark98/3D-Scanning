import open3d as o3d
import copy 
from modern_robotics import *

down_voxel_size = 0.1
icp_distance = down_voxel_size * 15
result_icp_distance = down_voxel_size * 1.5

def cal_angle(pl_norm, R_dir):
    angle_in_radians = \
        np.arccos(
            np.abs(pl_norm.x*R_dir[0]+ pl_norm.y*R_dir[1] + pl_norm.z*R_dir[2])
            )

    return angle_in_radians

def registerLocalCloud(target, source):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        source_temp.voxel_down_sample(down_voxel_size)
        target_temp.voxel_down_sample(down_voxel_size)

        source_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        target_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        
        # source_temp.estimate_normals()
        # target_temp.estimate_normals()

        
        current_transformation = np.identity(4)
        
        result_icp_p2l = o3d.pipelines.registration.registration_icp(source_temp, target_temp, icp_distance,
                current_transformation, o3d.pipelines.registration.TransformationEstimationPointToPlane())

        print("----------------")
        print("initial guess from Point-to-plane ICP registeration")
        print(result_icp_p2l)
        print(result_icp_p2l.transformation)

        p2l_init_trans_guess = result_icp_p2l.transformation
        

        result_icp = o3d.pipelines.registration.registration_icp(source_temp, target_temp, result_icp_distance,
                p2l_init_trans_guess, o3d.pipelines.registration.TransformationEstimationPointToPlane())

        print("----------------")
        print("result icp")
        print(result_icp)
        print(result_icp.transformation)

        return result_icp.transformation


if __name__ == '__main__':

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        # Load point clouds
        pcds = []
        for i in range(48):
            pcd = o3d.io.read_point_cloud(f"./360degree_pointclouds/test_pointcloud_{i}.pcd")
            print(np.mean(np.asarray(pcd.points)[:, 2]))
            pcds.append(pcd)

        # Visualize the mesh
        o3d.visualization.draw_geometries(pcds)

        cloud_base = pcds[0]

        cloud1 = copy.deepcopy(cloud_base)

        detectTransLoop = np.identity(4)
        posWorldTrans = np.identity(4)

        for cloud2 in pcds[1:]:
            posLocalTrans = registerLocalCloud(cloud1, cloud2)

            detectTransLoop = np.dot(posLocalTrans, detectTransLoop)

            posWorldTrans =  np.dot(posWorldTrans, posLocalTrans)

            cloud1 = copy.deepcopy(cloud2)
            cloud2.transform(posWorldTrans)
            
            cloud_base = cloud_base + cloud2

            # downsampling
            cloud_base.voxel_down_sample(down_voxel_size)
            
            
        
        # Visualize the mesh
        o3d.visualization.draw_geometries([cloud_base])