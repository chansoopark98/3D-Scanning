import open3d as o3d
import copy 
from modern_robotics import *
import cv2
from sfm.pose_utils import gen_poses

if __name__ == '__main__':
    pcds = []
    camera_poses = gen_poses('./frames/', 'exhaustive_matcher')
    
    len_poses = len(camera_poses)