import numpy as np
from record3d import Record3DStream
import cv2
from threading import Event
import open3d as o3d



class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

    def create_point_cloud(self, rgb, depth, intrinsics):
        # Convert the numpy arrays to Open3D Image format
        o3d_rgb_image = o3d.geometry.Image(rgb)
        o3d_depth_image = o3d.geometry.Image(depth)

        # Create an RGBDImage from the depth and RGB images
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb_image, o3d_depth_image, depth_scale=1000.0, depth_trunc=10.0, convert_rgb_to_intensity=False)

        # Create a point cloud from the RGBDImage and camera intrinsics
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

        return pcd


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

    def start_processing_stream(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        counter = 0
        prev_pcd = None
        while True:
            self.event.wait()  # Wait for new frame to arrive

            # Copy the newly arrived RGBD frame
            depth = self.session.get_depth_frame().astype(np.float32)
            depth = cv2.resize(depth, (720, 960))
            rgb = self.session.get_rgb_frame().astype(np.uint8)
            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
            intrinsics = o3d.camera.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2])
            camera_pose = self.session.get_camera_pose()  # Quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])

            camera_location = np.array([camera_pose.tx, camera_pose.ty, camera_pose.tz])  # Extract camera location

            print(f"Camera location: {camera_location}")


            # Postprocess it
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow('RGB', rgb)
            cv2.imshow('Depth', depth)

            try:
                # Create the point cloud using the depth and RGB images
                pcd = self.create_point_cloud(rgb, depth, intrinsics)
                if prev_pcd is not None:
                    # Register the current point cloud with the previous point cloud using the ICP algorithm
                    transformation_init = np.identity(4)
                    evaluation = o3d.pipelines.registration_fast_based_on_feature_matching(pcd, prev_pcd, 0.05, transformation_init)
                    print("Initial alignment\n", evaluation)

                    # Apply point-to-point ICP
                    reg_p2p = o3d.pipelines.registration_icp(pcd, prev_pcd, 0.05, transformation_init, o3d.pipelines.TransformationEstimationPointToPoint())
                    transformation = reg_p2p.transformation

                    # Transform the current point cloud using the computed transformation
                    pcd.transform(transformation)

                # Merge the transformed point cloud with the accumulated point cloud
                # Check if accumulated_pcd is defined
                if 'accumulated_pcd' not in locals():
                    accumulated_pcd = pcd
                else:
                    accumulated_pcd += pcd
                
                # Update the previous point cloud
                prev_pcd = pcd

                # Create a mesh from the accumulated point cloud using Poisson surface reconstruction
                accumulated_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(accumulated_pcd, depth=8)

                # Visualize the mesh
                vis.clear_geometries()
                vis.add_geometry(mesh)
                vis.poll_events()
                vis.update_renderer()
            except Exception as e:
                print(depth.shape)
                print(rgb.shape)
                # print numpy type
                print(type(depth))
                print(type(rgb))
                # Print the exception
                print(f'Exception: {e}')
            cv2.waitKey(1)
            self.event.clear()
            counter += 1

        vis.destroy_window()



if __name__ == '__main__':
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()