import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import copy
import cv2

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
    rgb_list = []
    depth_list = []
    capture_idx = 24

    # Capture
    capture = k4a.get_capture()
    rgb = capture.color

    # select roit
    x, y, w, h = cv2.selectROI(rgb)

    # pointcloud roi mask
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    mask = mask.astype(np.int16)
    indicies = np.where(mask==1)
    # mask = np.expand_dims(mask, axis=-1)
    
    cv2.destroyAllWindows()
    time.sleep(0.5)

    while True:
        idx += 1
        if idx == capture_idx + 1:
                break

        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        rgb = capture.color
        depth = capture.transformed_depth
    
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, :, :3].astype(np.uint8)
        print(rgb.shape)

        depth = depth * mask
        rgb = rgb * np.expand_dims(mask, axis=-1)

        gray = rgb.copy().astype(np.uint8)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        
        canny = cv2.Canny(gray, threshold1=200, threshold2=255)
        cv2.imshow('test', canny)
        cv2.waitKey(0)

        # max_range_mask = np.where(np.logical_and(depth<650, depth>400), 1, 0)
        # depth = depth * max_range_mask
        # rgb = rgb * np.expand_dims(max_range_mask, axis=-1)
        # print(depth.shape)
        # rgb_list.append(rgb)
        # depth_list.append(depth)

        # time.sleep(1)