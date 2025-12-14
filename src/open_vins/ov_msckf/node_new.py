#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import zmq
import cv2
import numpy as np

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import message_filters


TOPIC_IMG  = "/alphasense/cam0/image_raw" 
TOPIC_ODOM = "ov_msckf/odomimu"


JPEG_QUALITY = 90
ZMQ_BIND = "tcp://*:5555"  
SLOP_TIME = 0.2  # 허용 오차 (초)

bridge = CvBridge()

def pack_odom(odom_msg):
    p = odom_msg.pose.pose.position
    q = odom_msg.pose.pose.orientation
    # [x,y,z,qx,qy,qz,qw]
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32).tobytes()

def debug_img_cb(msg):

    #print("[RAW] Image Recv: {:.3f}".format(msg.header.stamp.to_sec()))
    pass

def debug_odom_cb(msg):

    # print("[RAW] Odom Recv: {:.3f}".format(msg.header.stamp.to_sec()))
    pass
# ----------------------

def main():
    rospy.init_node("mast3r_integrated_bridge", anonymous=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(ZMQ_BIND)

    rospy.Subscriber(TOPIC_IMG, Image, debug_img_cb)
    rospy.Subscriber(TOPIC_ODOM, Odometry, debug_odom_cb)

    img_sub  = message_filters.Subscriber(TOPIC_IMG, Image)
    odom_sub = message_filters.Subscriber(TOPIC_ODOM, Odometry)

    sync = message_filters.ApproximateTimeSynchronizer(
        [img_sub, odom_sub], queue_size=50, slop=SLOP_TIME
    )

    def bridge_cb(img_msg, odom_msg):
        #print("[Bridge] SYNC SUCCESS! TS: {:.3f}".format(img_msg.header.stamp.to_sec()))
        
        try:
            img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")

            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                return

            stamp = img_msg.header.stamp.to_sec()
            stamp_bytes = np.float64(stamp).tobytes()
            odom_bytes  = pack_odom(odom_msg)

            #  (topic | timestamp | pose | jpg)
            sock.send_multipart([b"kf", stamp_bytes, odom_bytes, buf.tobytes()])
            
        except Exception as e:
            rospy.logerr("Bridge Error: %s", str(e))

    sync.registerCallback(bridge_cb)

    print("---------------------------------------------------------")
    print(" Integrated Bridge Started!")
    print("  - Image Topic: {}".format(TOPIC_IMG))
    print("  - Odom  Topic: {}".format(TOPIC_ODOM))
    print("  - ZMQ Bind:    {}".format(ZMQ_BIND))
    print("---------------------------------------------------------")
    print("WAITING FOR DATA... (If no log appears, check ROS topics)")
    
    try:
        rospy.spin()
    finally:
        sock.close()
        ctx.term()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass