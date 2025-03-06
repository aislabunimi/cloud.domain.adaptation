#!/usr/bin/python3
# license removed for brevity
import rospy
from std_msgs.msg import String, Header
import cv2
import os
import tf2_ros
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import tf_conversions
import copy

def get_camera_info(scannet_folder, scene_id, W_unscaled, H_unscaled, W, H, type):
    K = np.loadtxt(os.path.join(scannet_folder, 'scans', scene_id, 'intrinsic', f'intrinsic_{type}.txt')).astype(float)
    
    scale_x = W / W_unscaled
    scale_y = H / H_unscaled

    K[0, 0] = K[0, 0] * scale_x  # fx
    K[1, 1] = K[1, 1] * scale_y  # fy
    K[0, 2] = K[0, 2] * scale_x  # cx
    K[1, 2] = K[1, 2] * scale_y  # cy

    return K 

def read_pose_file(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    pose_matrix = []
    for l in lines:
        line = l.rstrip('\n').split(' ')
        line = np.array(line).astype(float).reshape(4).tolist()
        pose_matrix.append(line)
    
    return np.array(pose_matrix).astype(float).reshape(4, 4)

def scale_image(image, W, H):

    H_original, W_original = image.shape[:2]
    
    
    if W_original != W or H_original != H:
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_NEAREST)
    
    return image

def publish_all_topics(scene_id, first_message_number, scannet_folder,
                       W_color, H_color,
                       W_depth, H_depth):
    
    publisher_image = rospy.Publisher('camera/color/image_raw', Image, queue_size=10)
    publisher_camera_info = rospy.Publisher('camera/color/camera_info', CameraInfo, queue_size=10)
    publisher_point_cloud = rospy.Publisher('camera/depth/points', PointCloud2, queue_size=10)
    
    rospy.init_node('scannet_to_ros')
    rate = rospy.Rate(30) # 30hz

    bridge = CvBridge()
    files = os.listdir(os.path.join(scannet_folder, 'scans', scene_id, 'color_scaled'))
    transform_broadcaster = tf2_ros.TransformBroadcaster()

    # Define camera info message
    W_unscaled_color = 1296
    H_unscaled_color = 968
    W_unscaled_depth = 640
    H_unscaled_depth = 480

    #     [fx  0 cx]
    # K = [ 0 fy cy]
    #     [ 0  0  1]
    K_color = get_camera_info(scannet_folder, scene_id, 
                                                W_unscaled=W_unscaled_color, H_unscaled=H_unscaled_color, 
                                                W=W_color, H=H_color, type='color') 
    K_depth = get_camera_info(scannet_folder, scene_id, 
                                                W_unscaled=W_unscaled_depth, H_unscaled=H_unscaled_depth, 
                                                W=W_depth, H=H_depth, type='depth') 

    while not rospy.is_shutdown():
        for num in range(len(files)):
            rospy.logerr((H_color, W_color))
            current_time = rospy.Time.now()
            # Camera color info message
            camera_info_color_message = CameraInfo()
            camera_info_color_message.header.stamp = current_time
            camera_info_color_message.width = W_color
            camera_info_color_message.height = H_color
            camera_info_color_message.K = K_color[0:3, 0:3].flatten()
            camera_info_color_message.header.frame_id = '/camera_rgb_link'

            # Camera depth info message
            camera_info_depth_message = CameraInfo()
            camera_info_depth_message.header.stamp = current_time
            camera_info_depth_message.width = W_depth
            camera_info_depth_message.height = H_depth
            camera_info_depth_message.K = K_depth[0:3, 0:3].flatten()
            camera_info_depth_message.header.frame_id = '/camera_depth_link'
            
            # Load pose
            pose_matrix = read_pose_file(os.path.join(scannet_folder, 'scans', scene_id, 'pose', f'{num}.txt'))
            camera_transform_message = TransformStamped() 
            camera_transform_message.header.stamp = current_time
            camera_transform_message.header.frame_id = '/map'
            camera_transform_message.child_frame_id = '/camera_link'
            camera_transform_message.transform.translation.x = pose_matrix[0, 3]
            camera_transform_message.transform.translation.y = pose_matrix[1, 3]
            camera_transform_message.transform.translation.z = pose_matrix[2, 3]

            rotation_quaternion = tf_conversions.transformations.quaternion_from_matrix(pose_matrix)
            camera_transform_message.transform.rotation.x = rotation_quaternion[0]
            camera_transform_message.transform.rotation.y = rotation_quaternion[1]
            camera_transform_message.transform.rotation.z = rotation_quaternion[2]
            camera_transform_message.transform.rotation.w = rotation_quaternion[3]

            # Load image 
            image_raw = cv2.imread(os.path.join(scannet_folder, 'scans', scene_id, 'color', f'{num}.jpg'))
            image_message_header = Header()
            image_message_header.stamp = current_time
            image_message_header.frame_id = 'camera_rgb_link'
            image_raw = scale_image(image_raw, W=W_color, H=H_color)
            image_message = bridge.cv2_to_imgmsg(image_raw, encoding='bgr8', header=image_message_header)

            # Load point cloud
            depth_raw = cv2.imread(os.path.join(scannet_folder, 'scans', scene_id, 'depth', f'{num}.png'), )

            # Publish all topics
            transform_broadcaster.sendTransform(camera_transform_message)
            publisher_image.publish(image_message)
            publisher_camera_info.publish(camera_info_color_message)
            


            rate.sleep()

if __name__ == '__main__':
    scene_id = rospy.get_param("scene_id", default='scene0000_00')
    first_message_number = rospy.get_param("first_message_number", default=0)
    scannet_folder = rospy.get_param("scannet_folder", default='/root/scannet')

    W_color = rospy.get_param("W_color", default=640)
    H_color = rospy.get_param("H_color", default=480)
    W_depth = rospy.get_param("W_depth", default=640)
    H_depth = rospy.get_param("W_depth", default=480)
    try:
        publish_all_topics(scene_id, first_message_number, scannet_folder, W_color, H_color, W_depth, H_depth)
    except rospy.ROSInterruptException:
        pass