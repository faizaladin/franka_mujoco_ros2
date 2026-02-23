import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class BoxDetector(Node):
    def __init__(self):
        super().__init__('box_detector')
        
        self.subscription = self.create_subscription(
            Image, 
            '/camera/rgb/image_raw', 
            self.image_callback, 
            10
        )
        self.bridge = CvBridge()
        self.get_logger().info("Looking for the Green Box (Rotated View)...")

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(c) > 100:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cx = x + w // 2
                cy = y + h // 2
                cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1)

        cv2.imshow("Original Feed", cv_image)
        cv2.imshow("Green Mask", mask)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = BoxDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
