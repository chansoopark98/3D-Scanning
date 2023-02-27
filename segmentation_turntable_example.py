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


    while cv2.waitKey(100) != ord('q'):
        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        rgb = capture.color
        depth = capture.transformed_depth
    
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, :, :3].astype(np.uint8)
        print(rgb.shape)

        depth = depth * mask
        masked_rgb = rgb.copy() * np.expand_dims(mask, axis=-1).astype(np.uint8)

        
        background = np.ones(masked_rgb.shape, dtype=np.uint8) * 255

        # # HSV 색 공간에서 녹색 영역을 검출하여 합성
        hsv = cv2.cvtColor(masked_rgb.copy(), cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (50, 150, 0), (70, 255, 255)) # 영상, 최솟값, 최댓값

        
        print(green_mask.shape)
        green_mask = cv2.add(background[:, :, 0], green_mask)
        

        cv2.copyTo(masked_rgb, green_mask, background)
        cv2.imshow('frame', green_mask)

    cv2.destroyAllWindows()