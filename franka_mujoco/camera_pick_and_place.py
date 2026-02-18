#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class CubeTracker(Node):
    def __init__(self):
        super().__init__('cube_tracker')
        self.bridge = CvBridge()
        
        self.sub_front_rgb = message_filters.Subscriber(self, Image, '/camera/front/image_raw')
        self.sub_front_depth = message_filters.Subscriber(self, Image, '/camera/front/depth_raw')
        self.sub_top_rgb = message_filters.Subscriber(self, Image, '/camera/top_down/image_raw')
        self.sub_top_depth = message_filters.Subscriber(self, Image, '/camera/top_down/depth_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_front_rgb, self.sub_front_depth, self.sub_top_rgb, self.sub_top_depth], 
            queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("Cube Tracker started. Calculating World Coordinates...")
        self.processed = False

    def sync_callback(self, f_rgb_msg, f_depth_msg, t_rgb_msg, t_depth_msg):
        if self.processed: return
        try:
            f_rgb = self.bridge.imgmsg_to_cv2(f_rgb_msg, 'bgr8')
            f_depth = self.bridge.imgmsg_to_cv2(f_depth_msg, 'passthrough')
            t_rgb = self.bridge.imgmsg_to_cv2(t_rgb_msg, 'bgr8')
            t_depth = self.bridge.imgmsg_to_cv2(t_depth_msg, 'passthrough')

            # XML Definitions
            cameras = [
                # Note: 'euler' in MuJoCo XML is Extrinsic XYZ (static frame)
                {'name': 'Front', 'rgb': f_rgb, 'depth': f_depth, 'pos': [1.6, 0.0, 1.8], 'euler': [0, 0.85, 1.57]},
                {'name': 'Top',   'rgb': t_rgb, 'depth': t_depth, 'pos': [0.5, 0.0, 1.2], 'euler': [0, 3.14, 0]}
            ]

            all_points = []
            all_colors = []

            for cam in cameras:
                pts, clrs = self.generate_cloud(cam['rgb'], cam['depth'], cam['pos'], cam['euler'])
                all_points.append(pts)
                all_colors.append(clrs)

            # Combine
            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors) / 255.0

            # Find Centers
            print("\n" + "="*40)
            # Red (High R, Low G/B), Green (Low R, High G, Low B)
            # XML GT: Red=[0.5, 0.3, 0.03], Green=[0.5, 0.0, 0.03]
            for color_name, gt in [('red', [0.5, 0.3, 0.03]), ('green', [0.5, 0.0, 0.03])]:
                center = self.get_centroid(combined_points, combined_colors, color_name)
                if center is not None:
                    print(f"{color_name.upper()} DETECTED: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
                    print(f"{color_name.upper()} GT (XML):  [{gt[0]:.4f}, {gt[1]:.4f}, {gt[2]:.4f}]")
                    err = np.linalg.norm(center - np.array(gt))
                    print(f"Error: {err:.4f} m (Expected ~0.015m offset due to box surface)")
                else:
                    print(f"{color_name.upper()} cube not found.")
            print("="*40 + "\n")

            # Save debug cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(combined_points)
            pcd.colors = o3d.utility.Vector3dVector(combined_colors)
            o3d.io.write_point_cloud("debug_world_coords.pcd", pcd)
            
            self.processed = True

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def get_centroid(self, points, colors, target_color):
        if target_color == 'red':
            mask = (colors[:, 0] > 0.6) & (colors[:, 1] < 0.3) & (colors[:, 2] < 0.3)
        elif target_color == 'green':
            mask = (colors[:, 0] < 0.3) & (colors[:, 1] > 0.6) & (colors[:, 2] < 0.3)
        
        filtered = points[mask]
        if len(filtered) < 10: return None
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        clean_points = np.asarray(pcd.points)[ind]
        if len(clean_points) == 0: return None
        return np.mean(clean_points, axis=0)

    def generate_cloud(self, rgb, depth, pos, euler):
        h, w = depth.shape
        # fov 50 degrees vertical
        f = 0.5 * h / np.tan(50 * np.pi / 360.0)
        cx, cy = w / 2.0, h / 2.0

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        mask = (depth > 0.1) & (depth < 4.0)
        z_opt = depth[mask]
        u_m = u[mask]
        v_m = v[mask]
        colors = rgb[mask][:, [2, 1, 0]]

        # 1. Optical Frame (OpenCV)
        # X=Right, Y=Down, Z=Forward
        x_opt = (u_m - cx) * z_opt / f
        y_opt = (v_m - cy) * z_opt / f
        
        # 2. MuJoCo Camera Frame
        # MuJoCo looks down -Z. Optical looks down +Z.
        # MuJoCo Up is +Y. Optical Down is +Y.
        # MuJoCo Right is +X. Optical Right is +X.
        # Mapping: [X_opt, -Y_opt, -Z_opt]
        cam_points_local = np.stack((x_opt, -y_opt, -z_opt), axis=-1)

        # 3. Transform to World
        # IMPORTANT: MuJoCo 'euler' is Extrinsic XYZ (Capital XYZ in Scipy)
        rot = R.from_euler('XYZ', euler) 
        
        world_points = rot.apply(cam_points_local) + np.array(pos)

        return world_points, colors

def main(args=None):
    rclpy.init(args=args)
    node = CubeTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()