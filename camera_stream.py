#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

def camera_stream():
   
    rospy.init_node('camera_stream_node')

    
    pub = rospy.Publisher('/camera_frames', Image, queue_size=10)

    bridge = CvBridge()

    
    source = rospy.get_param('~camera_source', '/home/bassant/downloads/people_walking.mp4')  # default path
    fps = rospy.get_param('~frame_rate', 10)  # desired publishing rate

    # Check if source is file or camera index
    if isinstance(source, str) and not os.path.isfile(source):
        rospy.logerr(f"Video file not found: {source}")
        return

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        rospy.logerr("Cannot open video source")
        return

    # Optional: use video's native FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps > 0:
        rospy.loginfo(f"Video FPS detected: {video_fps}")
        fps = min(fps, video_fps)  # don't exceed video FPS

    rate = rospy.Rate(fps)

    rospy.loginfo("Starting video stream...")

    while not rospy.is_shutdown():
        ret, frame = cap.read()

        if not ret:
            # Reached end of video, reset to beginning
            rospy.logwarn("Reached end of video, restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                rospy.logerr("Failed to read video after reset")
                break

        # Publish frame to ROS topic
        msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        pub.publish(msg)

        rate.sleep()

    cap.release()
    rospy.loginfo("Video stream node stopped.")

if __name__ == '__main__':
    try:
        camera_stream()
    except rospy.ROSInterruptException:
        pass
