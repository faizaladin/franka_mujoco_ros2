import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PointStamped
import numpy as np
import math

class ForwardKinematics(Node):
    def __init__(self):
        super().__init__('forward_kin')
        
        self.sub_joints = self.create_subscription(JointState, '/joint_states', self.forward, 10)
        self.sub_end_effector = self.create_subscription(PointStamped, '/end_effector', self.get_end_effector, 10)

        self.actual_pos = None

        self.get_logger().info("FK Node listening...")

        # DH Parameters [a, d, alpha, offset]
        # Correct Standard DH for Franka Emika Panda
        self.dh_params = [
            # Joint 1: Vertical to Horizontal (Twist -90)
            [0,      0.333, -math.pi/2, 0],
            # Joint 2: Horizontal to Vertical (Twist +90)
            [0,      0,      math.pi/2, 0],
            # Joint 3: Vertical to Horizontal
            [0.0825, 0.316,  math.pi/2, 0],
            # Joint 4: Horizontal to Vertical
            [-0.0825, 0,    -math.pi/2, 0],
            # Joint 5: Vertical to Horizontal
            [0,      0.384,  math.pi/2, 0],
            # Joint 6: Horizontal to Vertical
            [0.088,  0,      math.pi/2, 0],
            # Joint 7: Hand Flange
            [0,      0.107,  0,         0]
        ]
    def get_transform_matrix(self, a, d, alpha, theta):
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)

        return np.array([
            [ct, -st*ca,  st*sa,  a*ct],
            [st,  ct*ca, -ct*sa,  a*st],
            [0,   sa,     ca,     d],
            [0,   0,      0,      1]
        ])
    
    def get_end_effector(self, msg):
        self.actual_pos = np.array([msg.point.x, msg.point.y, msg.point.z])

    def forward(self, msg):
        pos = np.array(msg.position)
        vel = np.array(msg.velocity)
        result = np.eye(4)
        for i in range(7):
            a, d, alpha, offset = self.dh_params[i]
            result = np.dot(result, self.get_transform_matrix(a, d, alpha, pos[i] + offset))

        pos_hand = np.eye(4)
        pos_hand[2, 3] = 0.107
        pos_final = np.dot(result, pos_hand)

        calc_pos = pos_final[:3, 3]

        # FIX 3: Compare with Actual (only if we have received truth data)
        if self.actual_pos is not None:
            error = np.linalg.norm(calc_pos - self.actual_pos)
            self.get_logger().info(
                f"\nCalc:   {np.round(calc_pos, 4)}\n"
                f"Actual: {np.round(self.actual_pos, 4)}\n"
                f"Error:  {error:.6f} m"
            )
        else:
            self.get_logger().info(f"Calc: {np.round(calc_pos, 4)} (Waiting for truth...)")



        # # 3. Log (Rounded to 3 decimals)
        # self.get_logger().info(f"POS: {np.round(pos, 3)}")
        # self.get_logger().info(f"VEL: {np.round(vel, 3)}")

def main(args=None):
    rclpy.init(args=args)
    node = ForwardKinematics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()