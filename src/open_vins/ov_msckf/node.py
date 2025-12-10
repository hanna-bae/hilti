#!/usr/bin/env python2
import rospy
import zmq
import cv2
import numpy as np

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import message_filters

TOPIC_IMG  = "/mast3r/keyframe/image"
TOPIC_ODOM = "/mast3r/keyframe/pose"

JPEG_QUALITY = 90
ZMQ_BIND = "tcp://*:5555"  
bridge = CvBridge()

def pack_odom(odom_msg):
    p = odom_msg.pose.pose.position
    q = odom_msg.pose.pose.orientation
    # [x,y,z,qx,qy,qz,qw]
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32).tobytes()

def main():
    rospy.init_node("mast3r_kf_relay_zmq", anonymous=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(ZMQ_BIND)

    img_sub  = message_filters.Subscriber(TOPIC_IMG, Image)
    odom_sub = message_filters.Subscriber(TOPIC_ODOM, Odometry)

    
    sync = message_filters.ApproximateTimeSynchronizer(
        [img_sub, odom_sub], queue_size=5, slop=0.1
    )

    def cb(img_msg, odom_msg):
        # Image -> cv2 (hilti: gray scale)
        rospy.loginfo("Callback Triggered! Sending to ZMQ...")
        img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return

        stamp = img_msg.header.stamp.to_sec()
        stamp_bytes = np.float64(stamp).tobytes()
        odom_bytes  = pack_odom(odom_msg)

        # topic | timestamp | pose | jpg
        sock.send_multipart([b"kf", stamp_bytes, odom_bytes, buf.tobytes()])

    sync.registerCallback(cb)

    rospy.loginfo("Relaying %s + %s  -> %s", TOPIC_IMG, TOPIC_ODOM, ZMQ_BIND)
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
