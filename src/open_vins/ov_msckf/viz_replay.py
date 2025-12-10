#!/usr/bin/env python
import rospy
import zmq
import numpy as np
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav_msgs.msg import Path

def main():
    rospy.init_node('mast3r_viz_relay')
    
    pose_pub = rospy.Publisher('/mast3r/current_pose', PoseStamped, queue_size=10)
    pub_raw = rospy.Publisher('/mast3r/raw_path', Path, queue_size=10)
    pub_opt = rospy.Publisher('/mast3r/opt_path', Path, queue_size=10)

    
    # ZMQ Subscriber 
    ctx = zmq.Context()
    sock_raw = ctx.socket(zmq.SUB)
    
    HOST_IP = "172.17.0.1" 
    sock_raw.connect("tcp://{}:5560".format(HOST_IP))
    sock_raw.setsockopt(zmq.SUBSCRIBE, b"")
    
    sock_opt = ctx.socket(zmq.SUB)
    sock_opt.connect("tcp://{}:5561".format(HOST_IP))
    sock_opt.setsockopt(zmq.SUBSCRIBE, b"")
    sock_opt.setsockopt(zmq.RCVTIMEO, 1)
    print("Relay Node Started. Listening on 5560 (Raw) & 5561 (Opt)...")
    
    path_msg = Path()
    path_msg.header.frame_id = "global"

    while not rospy.is_shutdown():
        try:
            raw_bytes = sock_raw.recv(zmq.NOBLOCK)
            data = np.frombuffer(raw_bytes, dtype=np.float64)
            
            tx, ty, tz, qx, qy, qz, qw = data
            
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "global"
            pose_msg.pose.position = Point(tx, ty, tz)
            pose_msg.pose.orientation = Quaternion(qx, qy, qz, qw)
            
            pose_pub.publish(pose_msg)
            
            path_msg.header.stamp = rospy.Time.now()
            path_msg.poses.append(pose_msg)
            pub_raw.publish(path_msg)
            
        except zmq.Again:
            pass

        try:           
            data_bytes = sock_opt.recv(zmq.NOBLOCK)
            data = np.frombuffer(data_bytes, dtype=np.float32)
            
            num_poses = len(data) // 7
            if num_poses > 0:
                path_msg_opt = Path()
                path_msg_opt.header.frame_id = "global"
                path_msg_opt.header.stamp = rospy.Time.now()

                for i in range(num_poses):
                    offset = i * 7
                    tx, ty, tz, qx, qy, qz, qw = data[offset:offset+7]
                    
                    pose_msg_opt = PoseStamped()
                    pose_msg_opt.header.stamp = rospy.Time.now()
                    pose_msg_opt.header.frame_id = "global"
                    pose_msg_opt.pose.position = Point(tx, ty, tz)
                    pose_msg_opt.pose.orientation = Quaternion(qx, qy, qz, qw)
                    
                    path_msg_opt.poses.append(pose_msg_opt)
           
            pub_opt.publish(path_msg)

        except zmq.Again:
            pass

        rospy.sleep(0.005)
if __name__ == '__main__':
    main()