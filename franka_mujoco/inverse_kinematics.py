import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64 
import numpy as np
import math

class FrankaAnalyticalIK:
    def __init__(self):
        # DH Constants for Franka Panda
        self.d1 = 0.3330
        self.d3 = 0.3160
        self.d5 = 0.3840
        self.d7e = 0.2104
        self.a4 = 0.0825
        self.a7 = 0.0880

        self.LL24 = 0.10666225
        self.LL46 = 0.15426225
        self.L24 = 0.326591870689
        self.L46 = 0.392762332715
        
        self.thetaH46 = 1.35916951803
        self.theta342 = 1.31542071191
        self.theta46H = 0.211626808766

        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    def solve(self, O_T_EE, q7):
        if q7 <= self.q_min[6] or q7 >= self.q_max[6]: return []

        R_EE = O_T_EE[0:3, 0:3]
        z_EE = O_T_EE[0:3, 2]
        p_EE = O_T_EE[0:3, 3]

        p_7 = p_EE - self.d7e * z_EE
        x_EE_6 = np.array([math.cos(q7 - math.pi/4), -math.sin(q7 - math.pi/4), 0.0])
        x_6 = R_EE @ x_EE_6
        
        norm_x6 = np.linalg.norm(x_6)
        if norm_x6 < 1e-6: return []
        x_6 = x_6 / norm_x6
        p_6 = p_7 - self.a7 * x_6

        p_2 = np.array([0.0, 0.0, self.d1])
        V26 = p_6 - p_2
        LL26 = np.dot(V26, V26)
        L26 = math.sqrt(LL26)

        if (self.L24 + self.L46 < L26) or (self.L24 + L26 < self.L46) or (L26 + self.L46 < self.L24):
            return [] 

        theta246 = math.acos(np.clip((self.LL24 + self.LL46 - LL26) / (2.0 * self.L24 * self.L46), -1.0, 1.0))
        q4 = theta246 + self.thetaH46 + self.theta342 - 2.0 * math.pi

        if not (self.q_min[3] <= q4 <= self.q_max[3]): return []

        theta462 = math.acos(np.clip((LL26 + self.LL46 - self.LL24) / (2.0 * L26 * self.L46), -1.0, 1.0))
        theta26H = self.theta46H + theta462
        D26 = -L26 * math.cos(theta26H)

        Z_6 = np.cross(z_EE, x_6)
        Y_6 = np.cross(Z_6, x_6)
        
        norm_y6 = np.linalg.norm(Y_6)
        norm_z6 = np.linalg.norm(Z_6)
        if norm_y6 < 1e-6 or norm_z6 < 1e-6: return []

        R_6 = np.zeros((3, 3))
        R_6[:, 0] = x_6
        R_6[:, 1] = Y_6 / norm_y6
        R_6[:, 2] = Z_6 / norm_z6
        
        V_6_62 = R_6.T @ (-V26)
        Phi6 = math.atan2(V_6_62[1], V_6_62[0])
        
        arg_asin = D26 / math.sqrt(V_6_62[0]**2 + V_6_62[1]**2)
        if abs(arg_asin) > 1.0: return []
        Theta6 = math.asin(arg_asin)

        q6_candidates = [math.pi - Theta6 - Phi6, Theta6 - Phi6]
        valid_solutions = []

        thetaP26 = 3.0 * math.pi / 2.0 - theta462 - theta246 - self.theta342
        thetaP = math.pi - thetaP26 - theta26H
        LP6 = L26 * math.sin(thetaP26) / math.sin(thetaP)

        for q6 in q6_candidates:
            while q6 <= self.q_min[5]: q6 += 2.0 * math.pi
            while q6 >= self.q_max[5]: q6 -= 2.0 * math.pi
            if not (self.q_min[5] <= q6 <= self.q_max[5]): continue

            z_6_5 = np.array([math.sin(q6), math.cos(q6), 0.0])
            z_5 = R_6 @ z_6_5
            V2P = p_6 - LP6 * z_5 - p_2
            L2P = np.linalg.norm(V2P)

            q1_base = math.atan2(V2P[1], V2P[0])
            q2_base = math.acos(np.clip(V2P[2] / L2P, -1.0, 1.0))

            shoulder_candidates = [(q1_base, q2_base), 
                                   (q1_base + math.pi if q1_base < 0 else q1_base - math.pi, -q2_base)]

            for (s_q1, s_q2) in shoulder_candidates:
                if not (self.q_min[0] <= s_q1 <= self.q_max[0]): continue
                if not (self.q_min[1] <= s_q2 <= self.q_max[1]): continue

                z_3 = V2P / L2P
                Y_3 = np.cross(-V26, V2P)
                if np.linalg.norm(Y_3) < 1e-6: continue 
                y_3 = Y_3 / np.linalg.norm(Y_3)
                x_3 = np.cross(y_3, z_3)

                c1, s1 = math.cos(s_q1), math.sin(s_q1)
                R_1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
                c2, s2 = math.cos(s_q2), math.sin(s_q2)
                R_1_2 = np.array([[c2, -s2, 0], [0, 0, 1], [-s2, -c2, 0]])
                R_2 = R_1 @ R_1_2
                x_2_3 = R_2.T @ x_3
                q3 = math.atan2(x_2_3[2], x_2_3[0])

                if not (self.q_min[2] <= q3 <= self.q_max[2]): continue

                VH4 = p_2 + self.d3 * z_3 + self.a4 * x_3 - p_6 + self.d5 * z_5
                c6_s, s6_s = math.cos(q6), math.sin(q6)
                R_5_6 = np.array([[c6_s, -s6_s, 0], [0, 0, -1], [s6_s, c6_s, 0]])
                R_5 = R_6 @ R_5_6.T
                V_5_H4 = R_5.T @ VH4
                q5 = -math.atan2(V_5_H4[1], V_5_H4[0])
                
                if not (self.q_min[4] <= q5 <= self.q_max[4]): continue
                valid_solutions.append([s_q1, s_q2, q3, q4, q5, q6, q7])

        return valid_solutions

class InverseKinematics(Node):
    def __init__(self):
        super().__init__('ik_node')
        self.sub_joints = self.create_subscription(JointState, '/joint_states', self.get_joint_states, 10)
        self.sub_target = self.create_subscription(PoseStamped, '/target_pose', self.get_target_pose, 10)
        self.sub_gripper = self.create_subscription(Float64, '/gripper_command', self.get_gripper_cmd, 10)
        
        self.pub_control = self.create_publisher(JointState, '/inverse_control', 10)
        
        self.joints_pos = None
        self.solver = FrankaAnalyticalIK()
        self.target_pos = None
        self.target_rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.gripper_width = 0.04 

        self.get_logger().info("IK Node Online (Single Gripper Joint Mode).")
        self.create_timer(0.05, self.inverse)

    def get_joint_states(self, msg):
        self.joints_pos = list(msg.position[:7])

    def get_target_pose(self, msg):
        self.target_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.get_logger().info(f"Target Received: {self.target_pos}")

    def get_gripper_cmd(self, msg):
        self.gripper_width = msg.data
        self.get_logger().info(f"Gripper Command: {self.gripper_width}")

    def inverse(self):
        if self.joints_pos is None or self.target_pos is None: return

        O_T_EE = np.eye(4)
        O_T_EE[0:3, 0:3] = self.target_rot_matrix
        O_T_EE[0:3, 3]   = self.target_pos

        current_q7 = self.joints_pos[6]
        q7_test_points = [current_q7, 0.0, 0.785, -0.785, 1.57, -1.57]
        
        all_solutions = []
        for q7 in q7_test_points:
            sols = self.solver.solve(O_T_EE, q7)
            if sols: all_solutions.extend(sols)

        if not all_solutions: return 

        best_sol = min(all_solutions, key=lambda s: np.linalg.norm(np.array(s) - np.array(self.joints_pos)))

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 
                    'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1']
        
        msg.position = best_sol + [self.gripper_width]
        self.pub_control.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematics()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
