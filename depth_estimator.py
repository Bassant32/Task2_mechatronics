#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
pub = None

def estimate_depth_fast(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Laplacian to estimate depth from focus/blur
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    depth = cv2.Laplacian(blur, cv2.CV_32F)
    depth = np.abs(depth)

    # Also use intensity as a depth cue (darker = farther)
    intensity = gray.astype(np.float32) / 255.0

    # Combine both cues
    depth_map = 0.7 * depth + 0.3 * intensity

    # Normalize to 0-1
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

    return depth_map.astype(np.float32)

def frame_callback(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.resize(frame, (320, 240))

        depth_map = estimate_depth_fast(frame)

        depth_msg = bridge.cv2_to_imgmsg(depth_map, encoding="32FC1")
        depth_msg.header = msg.header
        pub.publish(depth_msg)

        rospy.loginfo_throttle(5, "Depth estimation running...")

    except Exception as e:
        rospy.logerr(f"Depth callback error: {e}")

def main():
    global pub

    rospy.init_node('depth_estimation_node')

    pub = rospy.Publisher('/object_depth', Image, queue_size=1)
    rospy.Subscriber('/camera_frames', Image, frame_callback, queue_size=1)

    rospy.loginfo("Depth estimation node ready! (lightweight mode)")
    rospy.spin()

if __name__ == '__main__':
    main()
