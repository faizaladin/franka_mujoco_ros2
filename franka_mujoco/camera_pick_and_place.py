#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import sys
import time

class CubeTracker(Node):
    def __init__(self, target_color='red'):
        super().__init__('cube_tracker')
        self.bridge = CvBridge()
        self.target_color = target_color.lower()
        
        self.pub_target = self.create_publisher(PoseStamped, '/target_pose', 10)
        self.pub_gripper = self.create_publisher(Float64, '/gripper_command', 10)

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

            center = self.get_centroid(combined_points, combined_colors, self.target_color)

            if center is not None:
                print(f"\nFOUND {self.target_color.upper()} BOX at: {center}")
                self.processed = True
                self.execute_pick_sequence(center)
            else:
                self.get_logger().warn(f"Could not find {self.target_color} box.")

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def execute_pick_sequence(self, center):
        # --- TUNING PARAMETERS ---
        hover_z = float(center[2]) + 0.20
        
        # LOWER GRASP Z
        # center[2] is box top (~0.05m). 
        # Adding 0.035 puts flange at 0.085m.
        # Since fingers are ~0.10m long, tips will go to -0.015m (just slightly into table)
        # This ensures a deep grasp.
        grasp_z = float(center[2]) + 0.035

        self.get_logger().info(f"Target Grasp Z: {grasp_z:.4f}")

        def send_pose(x, y, z):
            msg = PoseStamped()
            msg.header.frame_id = "world"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.position.z = float(z)
            msg.pose.orientation.w = 1.0 
            self.pub_target.publish(msg)
        
        def send_gripper(width):
            msg = Float64()
            msg.data = width
            self.pub_gripper.publish(msg)

        def smooth_approach(start_z, end_z, steps=60, delay=0.05):
            z_steps = np.linspace(start_z, end_z, steps)
            for z in z_steps:
                send_pose(center[0], center[1], z)
                time.sleep(delay)

        # 2. EXECUTION SEQUENCE
        self.get_logger().info("--- 1. MOVE TO HOVER ---")
        send_gripper(0.04) 
        send_pose(center[0], center[1], hover_z)
        time.sleep(3.0) 

        self.get_logger().info("--- 2. SLOW APPROACH ---")
        smooth_approach(hover_z, grasp_z, steps=70, delay=0.05) 
        
        # CRITICAL UPDATE: Increased Settling Time
        self.get_logger().info(f"--- Settling (5s) ---")
        time.sleep(5.0) 

        self.get_logger().info("--- 3. GRASP ---")
        send_gripper(0.00) 
        time.sleep(2.0)    

        self.get_logger().info("--- 4. SLOW LIFT ---")
        smooth_approach(grasp_z, hover_z, steps=50, delay=0.05)
        
        self.get_logger().info("--- DONE ---")

    def get_centroid(self, points, colors, target_color):
        if target_color == 'red':
            mask = (colors[:, 0] > 0.6) & (colors[:, 1] < 0.3) & (colors[:, 2] < 0.3)
        elif target_color == 'green':
            mask = (colors[:, 0] < 0.3) & (colors[:, 1] > 0.6) & (colors[:, 2] < 0.3)
        else:
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
    target = 'red'
    for arg in sys.argv:
        clean_arg = arg.lower().strip('-')
        if clean_arg == 'green': target = 'green'
        elif clean_arg == 'red': target = 'red'

    node = CubeTracker(target_color=target)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()