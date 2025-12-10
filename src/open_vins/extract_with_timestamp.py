
# -*- coding: utf-8 -*-
import rosbag
import cv2
from cv_bridge import CvBridge
import os
import sys

# --- CONFIGURATION ---
bag_file = './data/exp02_construction_multilevel.bag'  
output_dir = './data/exp02/'
target_topic = '/alphasense/cam0/image_raw' 
# ---------------------
SKIP_STEP = 1
# ----------------------------

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

bridge = CvBridge()
total_count = 0 
saved_count = 0  

print("Reading " + bag_file + " with downsampling (Step: " + str(SKIP_STEP) + ")...")

try:
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[target_topic]):
            total_count += 1
            
            if (total_count % SKIP_STEP) != 0:
                continue

            try:
                # Convert ROS image to OpenCV format
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    
                
                # Timestamp preserves exact sensor time (Critical for SLAM)
                timestamp = msg.header.stamp.to_nsec()
                image_name = str(timestamp) + ".png"
                
                cv2.imwrite(os.path.join(output_dir, image_name), cv_img)
                
                saved_count += 1
                if saved_count % 100 == 0:
                    print("Saved " + str(saved_count) + " images (Processed " + str(total_count) + ")...")
                    
            except Exception as e:
                print("Error extracting frame: " + str(e))

    print("Done!") 
    print("Total processed: " + str(total_count))
    print("Total saved: " + str(saved_count) + " to '" + output_dir + "'")

except Exception as e:
    print("Error opening bag file: " + str(e))