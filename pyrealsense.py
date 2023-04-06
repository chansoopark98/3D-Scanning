import pyrealsense2 as rs
import numpy as np
import cv2
from enum import IntEnum

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

def get_profiles():
    ctx = rs.context()
    devices = ctx.query_devices()

    color_profiles = []
    depth_profiles = []
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print('Sensor: {}, {}'.format(name, serial))
        print('Supported video formats:')
        for sensor in device.query_sensors():
            for stream_profile in sensor.get_stream_profiles():
                stream_type = str(stream_profile.stream_type())

                if stream_type in ['stream.color', 'stream.depth']:
                    v_profile = stream_profile.as_video_stream_profile()
                    fmt = stream_profile.format()
                    w, h = v_profile.width(), v_profile.height()
                    fps = v_profile.fps()

                    video_type = stream_type.split('.')[-1]
                    print('  {}: width={}, height={}, fps={}, fmt={}'.format(
                        video_type, w, h, fps, fmt))
                    if video_type == 'color':
                        color_profiles.append((w, h, fps, fmt))
                    else:
                        depth_profiles.append((w, h, fps, fmt))

    return color_profiles, depth_profiles

class PyRealSenseCamera(object):
    def __init__(self):
        # Configure the streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        print(self.config)
        """
        [Open3D INFO]   depth_fps: [100 | 15 | 30 | 300 | 6 | 60 | 90]
        [Open3D INFO]   depth_resolution: [1280,720 | 256,144 | 424,240 | 480,270 | 640,360 | 640,480 | 848,100 | 848,480]
        [Open3D INFO]   depth_format: [RS2_FORMAT_Z16]
        [Open3D INFO]   color_fps: [15 | 30 | 6 | 60]
        [Open3D INFO]   visual_preset: []
        [Open3D INFO]   color_resolution: [1280,720 | 1920,1080 | 320,180 | 320,240 | 424,240 | 640,360 | 640,480 | 848,480 | 960,540]
        [Open3D INFO]   color_format: [RS2_FORMAT_BGR8 | RS2_FORMAT_BGRA8 | RS2_FORMAT_RGB8 | RS2_FORMAT_RGBA8 | RS2_FORMAT_Y16 | RS2_FORMAT_YUYV]
        """
        self.config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
        # self.config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
        # self.config.enable_stream(rs.stream.color, 1280,720, rs.format.rgb8, 30)

        # Start the stream
        self.profile = self.pipeline.start(self.config)

        # Get the depth sensor's depth scale
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # Create an align object
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.color_frame = None
        self.depth_frame = None

        self.sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        self.sensor.set_option(rs.option.exposure, 156.000)

        self.capture()

    def get_profile(self):
        color_profiles, depth_profiles = get_profiles()
        print('Using the default profiles: \n  color:{}, depth:{}'.format(
            color_profiles[0], depth_profiles[0]))
        print('depth_profiles', depth_profiles[0])
        print('color_profiles', color_profiles[0])

    def capture(self):
        self.frames = self.pipeline.wait_for_frames()
        self.aligned_frames = self.align.process(self.frames)
        self.color_frame = self.aligned_frames.get_color_frame()
        self.depth_frame = self.aligned_frames.get_depth_frame()

    def get_color(self) -> np.ndarray:
        self.color_image = np.asanyarray(self.color_frame.get_data()).astype('uint8')
        return self.color_image

    def get_depth(self) -> np.ndarray:
        self.depth_image = np.asanyarray(self.depth_frame.get_data()).astype('uint16')
        return self.depth_image
    
    def get_camera_intrinsic(self) -> np.ndarray:
        if self.color_frame == None:
            self.capture()
            _ = self.get_color()

        color_intrinsics = self.color_frame.profile.as_video_stream_profile().intrinsics

        K = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ])
        return K

if __name__ == '__main__':
    camera = PyRealSenseCamera()
    camera.get_profile()
    while True:
        camera.capture()
        rgb = camera.get_color()
        depth = camera.get_depth()
        expand_depth = np.expand_dims(depth, axis=-1)
        rgb = np.where(expand_depth==0, 127, rgb) 
        
        cv2.imshow('test', rgb)
        cv2.waitKey(1)