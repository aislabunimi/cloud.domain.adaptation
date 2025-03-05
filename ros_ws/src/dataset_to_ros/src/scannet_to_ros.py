#!/usr/bin/python3
# license removed for brevity
import rospy
from std_msgs.msg import String, Header
import cv2
import os
import tf2_ros
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import tf_conversions

def read_pose_file(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    pose_matrix = []
    for l in lines:
        line = l.rstrip('\n').split(' ')
        line = np.array(line).astype(float).reshape(4).tolist()
        pose_matrix.append(line)
    
    return np.array(pose_matrix).astype(float).reshape(4, 4)

def publish_all_topics(scene_id, first_message_number, scannet_folder):
    
    publisher_image = rospy.Publisher('color_image', Image, queue_size=10)
    rospy.init_node('scannet_to_ros')
    rate = rospy.Rate(30) # 10hz

    bridge = CvBridge()
    files = os.listdir(os.path.join(scannet_folder, 'scans', scene_id, 'color_scaled'))
    transform_broadcaster = tf2_ros.TransformBroadcaster()
    while not rospy.is_shutdown():
        for num in range(len(files)):
            
            current_time = rospy.Time.now()
            
            # Load pose
            pose_matrix = read_pose_file(os.path.join(scannet_folder, 'scans', scene_id, 'pose', f'{num}.txt'))
            camera_transform_message = TransformStamped() 
            camera_transform_message.header.stamp = current_time
            camera_transform_message.header.frame_id = '/world'
            camera_transform_message.child_frame_id = '/camera'
            camera_transform_message.transform.translation.x = pose_matrix[0, 3]
            camera_transform_message.transform.translation.y = pose_matrix[1, 3]
            camera_transform_message.transform.translation.z = pose_matrix[2, 3]

            rotation_quaternion = tf_conversions.transformations.quaternion_from_matrix(pose_matrix)
            camera_transform_message.transform.rotation.x = rotation_quaternion[0]
            camera_transform_message.transform.rotation.y = rotation_quaternion[1]
            camera_transform_message.transform.rotation.z = rotation_quaternion[2]
            camera_transform_message.transform.rotation.w = rotation_quaternion[3]

            # Load image 
            image_raw = cv2.imread(os.path.join(scannet_folder, 'scans', scene_id, 'color_scaled', f'{num}.jpg'))
            image_message_header = Header()
            image_message_header.stamp = current_time
            image_message_header.frame_id = 'color_frame'
            image_message = bridge.cv2_to_imgmsg(image_raw, encoding='bgr8', header=image_message_header)

            # Publish all topics
            transform_broadcaster.sendTransform(camera_transform_message)
            publisher_image.publish(image_message)


            rate.sleep()

if __name__ == '__main__':
    scene_id = rospy.get_param("/scene_id", default='scene0000_00')
    first_message_number = rospy.get_param("/first_message_number", default=0)
    scannet_folder = rospy.get_param("scannet_folder", default='/root/scannet')
    try:
        publish_all_topics(scene_id, first_message_number, scannet_folder)
    except rospy.ROSInterruptException:
        pass