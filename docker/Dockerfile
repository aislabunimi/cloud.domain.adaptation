FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# Setup nvidia variables
RUN echo "export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY}}" >> ~/.bashrc
# Install ROS Noetic desktop full
# Minimal setup
RUN apt-get update \
 && apt-get install -y locales lsb-release
ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales
 
# Install ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update \
 && apt-get install -y --no-install-recommends ros-noetic-desktop-full
RUN apt-get install -y --no-install-recommends python3-rosdep build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools 
RUN apt-get install -y --no-install-recommends git
RUN rosdep init \
 && rosdep fix-permissions \
 && rosdep update

# Source ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# # Ros packages
# RUN apt-get update && apt-get install -y ros-noetic-rosserial-server
RUN apt-get update && apt-get install -y ros-noetic-code-coverage


# #Python packages
RUN apt install -y python3-pip 
RUN pip3 install -U catkin_tools 

# Build and source our workspace
RUN catkin_make --directory /root/code/ros_ws
RUN echo "source /root/code/ros_ws/devel/setup.bash" >> ~/.bashrc

RUN echo "Build complete"