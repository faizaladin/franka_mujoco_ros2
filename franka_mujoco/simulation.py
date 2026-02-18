import time
import os
import mujoco
import mujoco.viewer
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

class FrankaSim(Node):
    def __init__(self):
        super().__init__('franka_sim')
        
        # --- Publishers for Front Camera ---
        self.front_rgb_pub = self.create_publisher(Image, '/camera/front/image_raw', 10)
        self.front_depth_pub = self.create_publisher(Image, '/camera/front/depth_raw', 10)
        
        # --- Publishers for Top-Down Camera ---
        self.top_rgb_pub = self.create_publisher(Image, '/camera/top_down/image_raw', 10)
        self.top_depth_pub = self.create_publisher(Image, '/camera/top_down/depth_raw', 10)

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.end_eff_pub = self.create_publisher(PointStamped, '/end_effector', 10)
        self.sub_control = self.create_subscription(JointState, '/inverse_control', self.get_control, 10)
        self.bridge = CvBridge()

        self.control = None
        pkg_share = get_package_share_directory('franka_mujoco')
        xml_path = os.path.join(pkg_share, 'franka_mujoco', 'assets', 'franka_emika_panda', 'mjx_single_cube.xml')
        
        self.get_logger().info(f"Loading Scene: {xml_path}")

        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        except ValueError as e:
            self.get_logger().error(f"Failed to load XML: {e}")
            raise e

        self.data = mujoco.MjData(self.model)
        
        self.has_camera = False
        if self.model.ncam > 0:
            # Shared renderer for all camera passes
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.has_camera = True
            self.get_logger().info(f"Initialized with {self.model.ncam} cameras.")
        else:
            self.get_logger().warn("No camera found.")

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.create_timer(0.01, self.timer_callback)

    def publish_camera(self):
        # Define camera configurations to loop through
        cam_configs = [
            {"name": "rgb_camera", "rgb_pub": self.front_rgb_pub, "depth_pub": self.front_depth_pub, "frame": "front_cam_link"},
            {"name": "top_down_camera", "rgb_pub": self.top_rgb_pub, "depth_pub": self.top_depth_pub, "frame": "top_down_link"}
        ]

        timestamp = self.get_clock().now().to_msg()

        for cam in cam_configs:
            # 1. Render RGB
            self.renderer.update_scene(self.data, camera=cam["name"])
            rgb_image = self.renderer.render()

            # 2. Render Depth
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera=cam["name"])
            depth_image = self.renderer.render()
            self.renderer.disable_depth_rendering()

            # 3. Convert and Publish RGB
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
            rgb_msg.header.stamp = timestamp
            rgb_msg.header.frame_id = cam["frame"]
            cam["rgb_pub"].publish(rgb_msg)

            # 4. Convert and Publish Depth
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='passthrough')
            depth_msg.header.stamp = timestamp
            depth_msg.header.frame_id = cam["frame"]
            cam["depth_pub"].publish(depth_msg)

    def timer_callback(self):
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        
        # Publish at 10Hz (every 10 steps if timer is 0.01s)
        if self.has_camera and int(self.data.time * 100) % 10 == 0:
            self.publish_camera()

        if int(self.data.time * 100) % 1 == 0:
            self.publish_joints()
            self.publish_end_effector()

        if self.control is not None and int(self.data.time * 100) % 10 == 0:
            self.data.ctrl[:8] = self.control[:8]

    # ... keep publish_joints, publish_end_effector, and get_control the same ...
    def publish_joints(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'finger_joint1', 'finger_joint2']
        msg.position = self.data.qpos[:9].tolist()
        msg.velocity = self.data.qvel[:9].tolist()
        self.joint_pub.publish(msg)

    def publish_end_effector(self):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        end_effector = self.data.body("hand").xpos
        msg.point.x = end_effector[0]
        msg.point.y = end_effector[1]
        msg.point.z = end_effector[2]
        self.end_eff_pub.publish(msg)
    
    def get_control(self, msg):
        self.control = np.array(msg.position)

def main(args=None):
    rclpy.init(args=args)
    node = FrankaSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'viewer'):
            node.viewer.close()
        node.destroy_node()
        rclpy.shutdown()