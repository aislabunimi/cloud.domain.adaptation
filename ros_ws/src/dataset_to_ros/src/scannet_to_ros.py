#!/usr/bin/python3
# license removed for brevity
import rospy
from std_msgs.msg import String, Header
import cv2
import os
import tf2_ros
import numpy as np
import sensor_msgs.point_cloud2 as point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
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

def create_point_cloud(depth_raw, W_depth, H_depth, K_depth, scale=1000):
    h, w = np.mgrid[0: H_depth, 0: W_depth]
    z = depth_raw / scale
    x = (w - K_depth[0, 2]) * z / K_depth[0, 0] # x = (w - cx) * z / fx
    y = (h - K_depth[1, 2]) * z / K_depth[1, 1] # y = (h - cy) * z / fy
    point_cloud_raw = np.dstack((x, y, z)).reshape(-1, 3)
    return point_cloud_raw

def publish_all_topics(scene_id, first_message_number, scannet_folder,
                       W_color, H_color,
                       W_depth, H_depth):
    
    publisher_image = rospy.Publisher('camera/color/image_raw', Image, queue_size=10)
    publisher_camera_color_info = rospy.Publisher('camera/color/camera_info', CameraInfo, queue_size=10)
    publisher_camera_depth_info = rospy.Publisher('camera/depth/camera_info', CameraInfo, queue_size=10)
    publisher_point_cloud = rospy.Publisher('camera/depth/points', PointCloud2, queue_size=10)

    rospy.init_node('scannet_to_ros')
    rate = rospy.Rate(30) # 30hz

    bridge = CvBridge()
    files = os.listdir(os.path.join(scannet_folder, 'scans', scene_id, 'color_scaled'))
    transform_broadcaster = tf2_ros.TransformBroadcaster()

    # Publish static transforms for camera_color_link and camera_depth_link
    static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster()
    #Depth
    transform_camera_depth_message = TransformStamped()
    #transform_camera_depth_message.header.stamp = rospy.Time.now()
    transform_camera_depth_message.header.frame_id = '/camera_link'
    transform_camera_depth_message.child_frame_id = '/camera_depth_link'
    transform_camera_depth_message.transform.rotation.x = 0
    transform_camera_depth_message.transform.rotation.y = 0
    transform_camera_depth_message.transform.rotation.z = 0
    transform_camera_depth_message.transform.rotation.w = 1
    # Color
    static_transform_broadcaster.sendTransform(transform_camera_depth_message)
    transform_camera_color_message = TransformStamped()
    #transform_camera_color_message.header.stamp = rospy.Time.now()
    transform_camera_color_message.header.frame_id = '/camera_link'
    transform_camera_color_message.child_frame_id = '/camera_color_link'
    transform_camera_color_message.transform.rotation.x = 0
    transform_camera_color_message.transform.rotation.y = 0
    transform_camera_color_message.transform.rotation.z = 0
    transform_camera_color_message.transform.rotation.w = 1
    
    #static_transform_broadcaster.sendTransform(transform_camera_color_message)


    # Define camera info message
    [H_unscaled_color, W_unscaled_color] = cv2.imread(os.path.join(scannet_folder, 'scans', scene_id, 'color', f'0.jpg')).shape[0:2]
    [H_unscaled_depth, W_unscaled_depth] = cv2.imread(os.path.join(scannet_folder, 'scans', scene_id, 'depth', f'0.png')).shape[0:2]

    #     [fx  0 cx]
    # K = [ 0 fy cy]
    #     [ 0  0  1]
    K_color = get_camera_info(scannet_folder, scene_id, 
                                                W_unscaled=W_unscaled_color, H_unscaled=H_unscaled_color, 
                                                W=W_color, H=H_color, type='color') 
    K_depth = get_camera_info(scannet_folder, scene_id, 
                                                W_unscaled=W_unscaled_depth, H_unscaled=H_unscaled_depth, 
                                                W=W_depth, H=H_depth, type='depth') 
    
    # Convert color image to depth dimension
    map_color, map_depth = cv2.initUndistortRectifyMap(
        np.array(K_color).reshape((4, 4))[0:3, 0:3],
        np.array([0, 0, 0, 0]),
        np.eye(3),
        np.array(K_depth).reshape((4, 4))[0:3, 0:3],
        (W_depth, H_depth),
        cv2.CV_32FC1,
      )

    while not rospy.is_shutdown():
        for num in range(len(files)):

            current_time = rospy.Time.now()
            # Camera color info message
            camera_info_color_message = CameraInfo()
            camera_info_color_message.header.stamp = current_time
            camera_info_color_message.width = W_color
            camera_info_color_message.height = H_color
            camera_info_color_message.K = K_color[0:3, 0:3].flatten()
            camera_info_color_message.R = K_color[:3, :3].flatten()
            camera_info_color_message.P = K_color[:3, :4].flatten()
            camera_info_color_message.distortion_model = "plumb_bob"
            camera_info_color_message.header.frame_id = '/camera_rgb_link'

            # Camera depth info message
            camera_info_depth_message = CameraInfo()
            camera_info_depth_message.header.stamp = current_time
            camera_info_depth_message.width = W_depth
            camera_info_depth_message.height = H_depth
            camera_info_depth_message.K = K_depth[0:3, 0:3].flatten().tolist()
            camera_info_depth_message.R = K_depth[:3, :3].flatten()
            camera_info_depth_message.P = K_depth[:3, :4].flatten()
            camera_info_depth_message.distortion_model = "plumb_bob"
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
            # Rescale images to depth size
            #image_raw = cv2.remap(
             #       image_raw,
             #       map_color,
             #       map_depth,
            #        interpolation=cv2.INTER_NEAREST,
             #       borderMode=cv2.BORDER_CONSTANT,
             #       borderValue=0,
             #   )
            image_raw = scale_image(image=image_raw, W=W_depth, H=H_depth)

            image_message = bridge.cv2_to_imgmsg(image_raw, encoding='bgr8', header=image_message_header)

            # Load point cloud
            depth_raw = cv2.imread(os.path.join(scannet_folder, 'scans', scene_id, 'depth', f'{num}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth_raw = scale_image(image=depth_raw, W=W_depth, H=H_depth)
            # Generate flattened point cloud
            # The raw data are organized as follow [x0, y0, z0, x1, y1, z1, ...]
            # This should be speicified in the point cloud message
            point_cloud_raw = create_point_cloud(depth_raw=depth_raw, W_depth=W_depth, H_depth=H_depth, K_depth=K_depth)
            point_cloud_message = PointCloud2()
            point_cloud_message.header.stamp = current_time
            point_cloud_message.header.frame_id = 'camera_depth_link'
            point_cloud_message.fields = [PointField('x', 0, PointField.FLOAT32, 1), 
                                          PointField('y', 4, PointField.FLOAT32, 1),
                                          PointField('z', 8, PointField.FLOAT32, 1)]
            point_cloud_message.height = 1
            point_cloud_message.width = len(point_cloud_raw)
            #point_cloud_message.is_dense = False,
            #oint_cloud_message.is_bigendian = False,
            point_cloud_message.point_step = 12*24
            point_cloud_message.row_step = 12*24 * len(point_cloud_raw)
            #point_cloud_message.data = point_cloud_raw.tobytes()
            point_cloud_message=point_cloud2.create_cloud(point_cloud_message.header, [PointField('x', 0, PointField.FLOAT32, 1), 
                                          PointField('y', 4, PointField.FLOAT32, 1),
                                          PointField('z', 8, PointField.FLOAT32, 1)], point_cloud_raw)
            #Data size (7372800 bytes) does not match width (307200) times height (1) times point_step (12).  Dropping message.

            


            # Publish all topics
            transform_broadcaster.sendTransform(camera_transform_message)
            publisher_image.publish(image_message)
            publisher_camera_color_info.publish(camera_info_color_message)
            publisher_camera_depth_info.publish(camera_info_depth_message)
            publisher_point_cloud.publish(point_cloud_message)
            rate.sleep()

if __name__ == '__main__':
    scene_id = rospy.get_param("/scannet_to_ros/scene_id", default='scene0000_00')
    first_message_number = rospy.get_param("/scannet_to_ros/first_message_number", default=0)
    scannet_folder = rospy.get_param("/scannet_to_ros/scannet_folder", default='/root/scannet')

    W_depth = rospy.get_param("/scannet_to_ros/W_depth", default=640)
    H_depth = rospy.get_param("/scannet_to_ros/H_depth", default=480)

    W_color = W_depth
    H_color = H_depth
    try:
        publish_all_topics(scene_id, first_message_number, scannet_folder, W_color, H_color, W_depth, H_depth)
    except rospy.ROSInterruptException:
        pass