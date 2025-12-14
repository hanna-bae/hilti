#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import message_filters
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

TOPIC_IMG  = "/alphasense/cam0/image_raw" 
TOPIC_ODOM = "ov_msckf/odomimu"

last_img_time = 0.0
last_odom_time = 0.0

def img_cb(msg):
    global last_img_time, last_odom_time
    last_img_time = msg.header.stamp.to_sec()
    check_diff("Image")

def odom_cb(msg):
    global last_img_time, last_odom_time
    last_odom_time = msg.header.stamp.to_sec()
    check_diff("Odom ")

def check_diff(source):
    if last_img_time == 0 or last_odom_time == 0:
        return

    diff = last_img_time - last_odom_time
    
    status = ""
    if abs(diff) < 0.1:
        status = "[PERFECT]"
    elif abs(diff) < 0.2:
        status = "[GOOD]"
    else:
        status = "[BAD - SYNC FAIL]"

    print("{} Received | Diff: {:.4f} sec | {}".format(source, diff, status))

def main():
    rospy.init_node("debug_time_diff", anonymous=True)
    
    rospy.Subscriber(TOPIC_IMG, Image, img_cb)
    rospy.Subscriber(TOPIC_ODOM, Odometry, odom_cb)
    
    print("------------------------------------------------")
    print(" Checking Time Difference betwen Image & Odom")
    print("------------------------------------------------")
    rospy.spin()

if __name__ == "__main__":
    main()