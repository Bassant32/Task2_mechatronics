#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import threading
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from PIL import Image as PILImage

bridge = CvBridge()
pub_depth = None
pub_distance = None

latest_frame = None
latest_header = None
frame_lock = threading.Lock()

processor = None
model = None

def estimate_distance(depth_map):
    threshold = np.percentile(depth_map, 80)
    close_region = depth_map[depth_map > threshold]
    avg_depth = float(np.mean(close_region))
    estimated_meters = (1.0 - avg_depth) * 10.0
    return round(estimated_meters, 2)

def frame_callback(msg):
    global latest_frame, latest_header
    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.resize(frame, (224, 224))
        with frame_lock:
            latest_frame = frame
            latest_header = msg.header
    except Exception as e:
        rospy.logerr(f"Frame callback error: {e}")

def depth_thread():
    global pub_depth, pub_distance
    rospy.loginfo("Depth processing thread started!")

    while not rospy.is_shutdown():
        with frame_lock:
            frame = latest_frame
            header = latest_header

        if frame is None:
            rospy.sleep(0.1)
            continue

        try:
            pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to('cuda') for k, v in inputs.items()}  # move inputs to GPU

            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            depth_map = predicted_depth.squeeze().numpy().astype(np.float32)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

            distance = estimate_distance(depth_map)

            depth_msg = bridge.cv2_to_imgmsg(depth_map, encoding="32FC1")
            depth_msg.header = header
            pub_depth.publish(depth_msg)
            pub_distance.publish(Float32(distance))

            rospy.loginfo_throttle(5, f"Depth running | Closest object: ~{distance}m")

        except Exception as e:
            rospy.logerr(f"Depth thread error: {e}")

def main():
    global pub_depth, pub_distance, processor, model

    rospy.init_node('depth_estimation_node')

    depth_model_path = rospy.get_param('~depth_model_path',
                                        'depth-anything/Depth-Anything-V2-Small-hf')
    rospy.loginfo(f"Loading DepthAnythingV2 model: {depth_model_path}")
    rospy.loginfo("This may take a minute on first run...")

    processor = AutoImageProcessor.from_pretrained(depth_model_path)
    model = AutoModelForDepthEstimation.from_pretrained(depth_model_path)
    model.eval()
    torch.set_num_threads(4)
    model = model.to('cuda')  # move model to GPU

    pub_depth = rospy.Publisher('/object_depth', Image, queue_size=1)
    pub_distance = rospy.Publisher('/object_distance', Float32, queue_size=1)

    rospy.Subscriber('/camera_frames', Image, frame_callback, queue_size=1)

    t = threading.Thread(target=depth_thread, daemon=True)
    t.start()

    rospy.loginfo("Depth estimation node ready!")
    rospy.spin()

if __name__ == '__main__':
    main()
