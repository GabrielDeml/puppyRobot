import numpy as np
from record3d import Record3DStream
import cv2
from threading import Event
import open3d as o3d
import sys
import select


class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        self.rgb_video_file = 'rgb_video.mp4'
        self.depth_video_file = 'depth_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'MJPG', 'H264', 'XVID'
        self.fps = 30  # You can set the desired frames per second for the output videos

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])
    
    def get_camera_location(self, camera_pose):
        # Extract the quaternion and translation vector from the camera pose
        qx, qy, qz, qw, tx, ty, tz = camera_pose.qx, camera_pose.qy, camera_pose.qz, camera_pose.qw, camera_pose.tx, camera_pose.ty, camera_pose.tz

        # Convert the quaternion to a rotation matrix
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion([qx, qy, qz, qw])

        # Create a transformation matrix from the rotation matrix and translation vector
        transformation_matrix = np.identity(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = [tx, ty, tz]

        return transformation_matrix

    def start_processing_stream(self):
        # Get the frame size of the input video
        self.event.wait()
        rgb = self.session.get_rgb_frame()
        depth = self.session.get_depth_frame()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Print the type
        print("RGB type:", type(rgb))
        frame_height_rgb = rgb.shape[0]
        frame_width_rgb = rgb.shape[1]
        frame_height_depth = depth.shape[0]
        frame_width_depth = depth.shape[1]
        print("Frame height RGB:", frame_height_rgb)
        print("Frame width RGB:", frame_width_rgb)
        print("Frame height depth:", frame_height_depth)
        print("Frame width depth:", frame_width_depth)

        # Create the VideoWriter objects
        rgb_out = cv2.VideoWriter(self.rgb_video_file, self.fourcc, self.fps, (frame_height_rgb, frame_width_rgb))

        if not rgb_out.isOpened():
            print("Error: Could not open RGB video file for writing")
            return
        depth_out = cv2.VideoWriter(self.depth_video_file, self.fourcc, self.fps, (frame_height_depth, frame_width_depth), isColor=False)

        if not depth_out.isOpened():
            print("Error: Could not open depth video file for writing")
            return
        try:
            while True:
                self.event.wait()  # Wait for new frame to arrive

                            # Copy the newly arrived RGBD frame
                depth = self.session.get_depth_frame()
                rgb = self.session.get_rgb_frame()
                intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
                camera_pose = self.session.get_camera_pose()  # Quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])
                print(intrinsic_mat)


                camera_location = self.get_camera_location(camera_pose)

                # Extract the x, y, and z coordinates from the camera location
                x, y, z = camera_location[:3, 3]

                # Map the camera's position to a 2D plane (ignoring the z-coordinate)
                camera_position_2d = (x, y)
                
                print("Camera position in 2D plane:", camera_position_2d)

                # print("Camera location (transformation matrix):\n", camera_location)


                # You can now e.g. create point cloud by projecting the depth map using the intrinsic matrix.

                # Postprocess it
                if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                    depth = cv2.flip(depth, 1)
                    rgb = cv2.flip(rgb, 1)

                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Save RGB frame to video
                rgb_out.write(rgb)

                depth_8u = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                # Save depth frame to video
                depth_out.write(depth_8u)


                self.event.clear()

        except KeyboardInterrupt:
            print("Exiting loop due to Ctrl+C")

        finally:
            # Release the video writer objects
            print("Releasing video writer objects")
            rgb_out.release()
            depth_out.release()



if __name__ == '__main__':
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()
