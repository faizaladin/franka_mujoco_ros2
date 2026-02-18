#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import sys # <--- NEW IMPORT

class CubeTracker(Node):
    def __init__(self, target_color='red'):
        super().__init__('cube_tracker')
        self.bridge = CvBridge()
        self.target_color = target_color.lower() # Store the target color
        
        self.pub_target = self.create_publisher(PoseStamped, '/target_pose', 10)

        self.sub_front_rgb = message_filters.Subscriber(self, Image, '/camera/front/image_raw')
        self.sub_front_depth = message_filters.Subscriber(self, Image, '/camera/front/depth_raw')
        self.sub_top_rgb = message_filters.Subscriber(self, Image, '/camera/top_down/image_raw')
        self.sub_top_depth = message_filters.Subscriber(self, Image, '/camera/top_down/depth_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_front_rgb, self.sub_front_depth, self.sub_top_rgb, self.sub_top_depth], 
            queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(f"Cube Tracker started. TARGETING: {self.target_color.upper()} BOX")
        self.processed = False

    def sync_callback(self, f_rgb_msg, f_depth_msg, t_rgb_msg, t_depth_msg):
        if self.processed: return
        try:
            f_rgb = self.bridge.imgmsg_to_cv2(f_rgb_msg, 'bgr8')
            f_depth = self.bridge.imgmsg_to_cv2(f_depth_msg, 'passthrough')
            t_rgb = self.bridge.imgmsg_to_cv2(t_rgb_msg, 'bgr8')
            t_depth = self.bridge.imgmsg_to_cv2(t_depth_msg, 'passthrough')

            cameras = [
                {'name': 'Front', 'rgb': f_rgb, 'depth': f_depth, 'pos': [1.6, 0.0, 1.8], 'euler': [0, 0.85, 1.57]},
                {'name': 'Top',   'rgb': t_rgb, 'depth': t_depth, 'pos': [0.5, 0.0, 1.2], 'euler': [0, 3.14, 0]}
            ]

            all_points = []
            all_colors = []

            for cam in cameras:
                pts, clrs = self.generate_cloud(cam['rgb'], cam['depth'], cam['pos'], cam['euler'])
                all_points.append(pts)
                all_colors.append(clrs)

            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors) / 255.0

            # --- Use the target color passed during init ---
            center = self.get_centroid(combined_points, combined_colors, self.target_color)

            if center is not None:
                print(f"\nFOUND {self.target_color.upper()} BOX at: {center}")
                
                msg = PoseStamped()
                msg.header.frame_id = "world"
                msg.header.stamp = self.get_clock().now().to_msg()
                
                msg.pose.position.x = float(center[0])
                msg.pose.position.y = float(center[1])
                msg.pose.position.z = float(center[2]) + 0.20 # Hover offset
                msg.pose.orientation.w = 1.0

                self.pub_target.publish(msg)
                self.get_logger().info(f"Published Target Pose to /target_pose: {msg.pose.position}")
                self.processed = True
            else:
                self.get_logger().warn(f"Could not find {self.target_color} box.")

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def get_centroid(self, points, colors, target_color):
        if target_color == 'red':
            mask = (colors[:, 0] > 0.6) & (colors[:, 1] < 0.3) & (colors[:, 2] < 0.3)
        elif target_color == 'green':
            mask = (colors[:, 0] < 0.3) & (colors[:, 1] > 0.6) & (colors[:, 2] < 0.3)
        else:
            self.get_logger().error(f"Unknown color: {target_color}")
            return None
        
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
        f = 0.5 * h / np.tan(50 * np.pi / 360.0)
        cx, cy = w / 2.0, h / 2.0

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        mask = (depth > 0.1) & (depth < 4.0)
        z_opt = depth[mask]
        u_m = u[mask]
        v_m = v[mask]
        colors = rgb[mask][:, [2, 1, 0]]

        x_opt = (u_m - cx) * z_opt / f
        y_opt = (v_m - cy) * z_opt / f
        
        cam_points_local = np.stack((x_opt, -y_opt, -z_opt), axis=-1)

        rot = R.from_euler('XYZ', euler)
        world_points = rot.apply(cam_points_local) + np.array(pos)

        return world_points, colors

def main(args=None):
    rclpy.init(args=args)
    
    # Parse arguments manually to avoid conflict with ros2 args
    target = 'red' # Default
    
    # Check if a color argument was passed
    # Usage: ros2 run package node -- red  OR ros2 run package node -- green
    for arg in sys.argv:
        if arg.lower() == 'green':
            target = 'green'
        elif arg.lower() == 'red':
            target = 'red'

    node = CubeTracker(target_color=target)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()