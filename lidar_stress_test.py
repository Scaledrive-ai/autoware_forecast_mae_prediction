import carla 
import math 
import random 
import time 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py.point_cloud2 import create_cloud
import time
from builtin_interfaces.msg import Time
from transforms3d.euler import euler2quat
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped

from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Pose
from autoware_vehicle_msgs.msg import VelocityReport, SteeringReport, ControlModeReport, GearReport
from tier4_vehicle_msgs.msg import ActuationStatusStamped
from rosgraph_msgs.msg import Clock
from queue import Queue, Empty
import tf_transformations as tft  # or use transforms3d if available
from transforms3d.euler import quat2euler


def carla_location_to_ros_point(carla_location):
    """Convert a carla location to a ROS point."""
    ros_point = Point()
    ros_point.x = carla_location.x
    ros_point.y = -carla_location.y
    ros_point.z = carla_location.z

    return ros_point


def carla_rotation_to_ros_quaternion(carla_rotation):
    """Convert a carla rotation to a ROS quaternion."""
    roll = math.radians(carla_rotation.roll)
    pitch = -math.radians(carla_rotation.pitch)
    yaw = -math.radians(carla_rotation.yaw)
    quat = euler2quat(roll, pitch, yaw)
    ros_quaternion = Quaternion(w=quat[0], x=quat[1], y=quat[2], z=quat[3])

    return ros_quaternion

def ros_quaternion_to_carla_rotation(ros_quaternion):
    """Convert ROS quaternion to carla rotation."""
    roll, pitch, yaw = quat2euler(
        [ros_quaternion.w, ros_quaternion.x, ros_quaternion.y, ros_quaternion.z]
    )

    return carla.Rotation(
        roll=math.degrees(roll), pitch=-math.degrees(pitch), yaw=-math.degrees(yaw)
    )

def ros_pose_to_carla_transform(ros_pose):
    """Convert ROS pose to carla transform."""
    return carla.Transform(
        carla.Location(ros_pose.position.x, -ros_pose.position.y, ros_pose.position.z),
        ros_quaternion_to_carla_rotation(ros_pose.orientation),
    )

sensor_data = {}



class CarlaLidarNode(Node):
    def __init__(self):
        super().__init__('carla_lidar_node')
        self.publisher1 = self.create_publisher(PointCloud2, '/sensing/lidar/top/pointcloud_before_sync', 10)
        self.publisher2 = self.create_publisher(PointCloud2, '/sensing/lidar/right/pointcloud_before_sync', 10)
        self.publisher3 = self.create_publisher(PointCloud2, '/sensing/lidar/left/pointcloud_before_sync', 10)
        self.imu_publisher = self.create_publisher(Imu, '/sensing/imu/tamagawa/imu_raw', 1)
        self.pose_publisher = self.create_publisher(PoseWithCovarianceStamped, "/sensing/gnss/pose_with_covariance", 1)

        self.pub_vel_state = self.create_publisher(VelocityReport, "/vehicle/status/velocity_status", 1)
        self.pub_steering_state = self.create_publisher(SteeringReport, "/vehicle/status/steering_status", 1)
        self.pub_ctrl_mode = self.create_publisher(ControlModeReport, "/vehicle/status/control_mode", 1)
        self.pub_gear_state = self.create_publisher(GearReport, "/vehicle/status/gear_status", 1)
        self.pub_actuation_status = self.create_publisher(ActuationStatusStamped, "/vehicle/status/actuation_status", 1)

        self.lidar_top_queue = Queue(maxsize=1)
        self.lidar_right_queue = Queue(maxsize=1)
        self.lidar_left_queue = Queue(maxsize=1)
        self.imu_queue = Queue(maxsize=1)
        self.pose_queue = Queue(maxsize=1)

        self.clock_publisher = self.create_publisher(Clock, "/clock", 10)
        #obj_clock = Clock()
        #obj_clock.clock = Time(sec=0)
        #self.clock_publisher.publish(obj_clock)

        self.sub_vehicle_initialpose = self.create_subscription(
            PoseWithCovarianceStamped, "initialpose", self.initialpose_callback, 1
        )
        self.ego_vehicle = None

    def initialpose_callback(self, data):
        """Transform RVIZ initial pose to CARLA."""
        pose = data.pose.pose
        pose.position.z += 2.0
        carla_pose_transform = ros_pose_to_carla_transform(pose)
        if self.ego_vehicle is not None:
            self.ego_vehicle.set_transform(carla_pose_transform)
            print("initializing pose")
        else:
            print("Can't find Ego Vehicle")

def main():
    # Setup CARLA client and world
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Set synchronous mode and 20Hz time step
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1 # 20Hz
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    spawn_point = world.get_map().get_spawn_points()
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point[59])

    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '2000000')
    lidar_bp.set_attribute('upper_fov', '10.0')
    lidar_bp.set_attribute('lower_fov', '-30.0')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('sensor_tick', '0.1')
    lidar_bp.set_attribute('noise_stddev', '0.0')
    lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=3.1), carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0))

    lidar1 = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_bp.set_attribute('points_per_second', '10000')
    lidar_transform = carla.Transform(carla.Location(x=3, y=3, z=10.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0))
    lidar2 = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_bp.set_attribute('points_per_second', '10000')
    lidar_transform = carla.Transform(carla.Location(x=-3, y=-3, z=10.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0))
    lidar3 = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute("noise_accel_stddev_x", str(0.0))
    imu_bp.set_attribute("noise_accel_stddev_y", str(0.0))
    imu_bp.set_attribute("noise_accel_stddev_z", str(0.0))
    imu_bp.set_attribute("noise_gyro_stddev_x", str(0.0))
    imu_bp.set_attribute("noise_gyro_stddev_y", str(0.0))
    imu_bp.set_attribute("noise_gyro_stddev_z", str(0.0))
    imu_transform = carla.Transform(carla.Location(x=0, y=0, z=1.6), carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0))
    imu = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)

    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss_bp.set_attribute("noise_alt_stddev", str(0.0))
    gnss_bp.set_attribute("noise_lat_stddev", str(0.0))
    gnss_bp.set_attribute("noise_lon_stddev", str(0.0))
    gnss_bp.set_attribute("noise_alt_bias", str(0.0))
    gnss_bp.set_attribute("noise_lat_bias", str(0.0))
    gnss_bp.set_attribute("noise_lon_bias", str(0.0))
    gnss_transform = carla.Transform(carla.Location(x=0, y=0, z=1.6), carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0))
    gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)

    ros_node = CarlaLidarNode()
    ros_node.ego_vehicle = vehicle



    def lidar_callback(carla_lidar_measurement, id, queue):
        '''
        # Structuring message as in autoware-carla interface
        pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
        intensity = (np.clip(pts[:, 3], 0, 1) * 255).astype(np.uint8).reshape(-1, 1)
        return_type = np.zeros((pts.shape[0], 1), dtype=np.uint8)
        channel = np.zeros((pts.shape[0], 1), dtype=np.uint16)    # Simplified test: all zero
        lidar_points = np.hstack((pts[:, :3], intensity, return_type, channel))

        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("intensity", "u1"), ("return_type", "u1"), ("channel", "u2"),
        ]
        structured = np.zeros(lidar_points.shape[0], dtype=dtype)
        structured['x'] = lidar_points[:, 0]
        structured['y'] = lidar_points[:, 1]
        structured['z'] = lidar_points[:, 2]
        structured['intensity'] = lidar_points[:, 3].astype(np.uint8)
        structured['return_type'] = lidar_points[:, 4].astype(np.uint8)
        structured['channel'] = lidar_points[:, 5].astype(np.uint16)

        # Prepare ROS2 PointCloud2 message
        header = Header()
        #header.stamp = ros_node.get_clock().now().to_msg()
        header.frame_id = id
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.UINT8, count=1),
            PointField(name="return_type", offset=13, datatype=PointField.UINT8, count=1),
            PointField(name="channel", offset=14, datatype=PointField.UINT16, count=1),
        ]
        msg = create_cloud(header, fields, structured)
        
        #lidar_publisher.publish(msg)
        #sensor_data[id] = msg
        '''

       
        
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.UINT8, count=1),
            PointField(name="return_type", offset=13, datatype=PointField.UINT8, count=1),
            PointField(name="channel", offset=14, datatype=PointField.UINT16, count=1),
        ]

        lidar_data = np.frombuffer(
            carla_lidar_measurement.raw_data, dtype=np.float32
        ).reshape(-1, 4)
        intensity = lidar_data[:, 3]
        intensity = (
            np.clip(intensity, 0, 1) * 255
        )  # CARLA lidar intensity values are between 0 and 1
        intensity = intensity.astype(np.uint8).reshape(-1, 1)

        return_type = np.zeros((lidar_data.shape[0], 1), dtype=np.uint8)
        channel = np.empty((0, 1), dtype=np.uint16)
        channels = 64

        for i in range(channels):
            current_ring_points_count = carla_lidar_measurement.get_point_count(i)
            channel = np.vstack(
                (channel, np.full((current_ring_points_count, 1), i, dtype=np.uint16))
            )

        lidar_data = np.hstack((lidar_data[:, :3], intensity, return_type, channel))
        lidar_data[:, 1] *= -1

        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("intensity", "u1"),
            ("return_type", "u1"),
            ("channel", "u2"),
        ]

        structured_lidar_data = np.zeros(lidar_data.shape[0], dtype=dtype)
        structured_lidar_data["x"] = lidar_data[:, 0]
        structured_lidar_data["y"] = lidar_data[:, 1]
        structured_lidar_data["z"] = lidar_data[:, 2]
        structured_lidar_data["intensity"] = lidar_data[:, 3].astype(np.uint8)
        structured_lidar_data["return_type"] = lidar_data[:, 4].astype(np.uint8)
        structured_lidar_data["channel"] = lidar_data[:, 5].astype(np.uint16)

        header = Header()
        header.frame_id = id

        point_cloud_msg = create_cloud(header, fields, structured_lidar_data)



        if queue.full():
            queue.get_nowait()
        queue.put(point_cloud_msg)

    def imu_callback(carla_imu_measurement, queue):
       
        imu_msg = Imu()
        #imu_msg.header.stamp = ros_node.get_clock().now().to_msg()
        imu_msg.header.frame_id = "tamagawa/imu_link_changed"

        imu_msg.angular_velocity.x = -carla_imu_measurement.gyroscope.x
        imu_msg.angular_velocity.y = carla_imu_measurement.gyroscope.y
        imu_msg.angular_velocity.z = -carla_imu_measurement.gyroscope.z

        imu_msg.linear_acceleration.x = carla_imu_measurement.accelerometer.x
        imu_msg.linear_acceleration.y = -carla_imu_measurement.accelerometer.y
        imu_msg.linear_acceleration.z = carla_imu_measurement.accelerometer.z

        roll = math.radians(carla_imu_measurement.transform.rotation.roll)
        pitch = -math.radians(carla_imu_measurement.transform.rotation.pitch)
        yaw = -math.radians(carla_imu_measurement.transform.rotation.yaw)

        quat = euler2quat(roll, pitch, yaw)
        imu_msg.orientation.w = quat[0]
        imu_msg.orientation.x = quat[1]
        imu_msg.orientation.y = quat[2]
        imu_msg.orientation.z = quat[3]

        #imu_publisher.publish(imu_msg)
        #sensor_data[f"imu"] = imu_msg

        if queue.full():
            queue.get_nowait()
        queue.put(imu_msg)

    def pose_callback(pose_data, queue):
        out_pose_with_cov = PoseWithCovarianceStamped()
        pose_carla = Pose()
        pose_carla.position = carla_location_to_ros_point(vehicle.get_transform().location)
        pose_carla.orientation = carla_rotation_to_ros_quaternion(
            vehicle.get_transform().rotation
        )
        header = Header()
        #header.stamp = ros_node.get_clock().now().to_msg()
        header.frame_id = 'map'
        out_pose_with_cov.header = header
        out_pose_with_cov.pose.pose = pose_carla
        out_pose_with_cov.pose.covariance = [
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        #pose_publisher.publish(out_pose_with_cov)
        #sensor_data[f"pose"] = out_pose_with_cov

        if queue.full():
            queue.get_nowait()
        queue.put(out_pose_with_cov)

        '''
        trans_mat = np.array(vehicle.get_transform().get_matrix()).reshape(4, 4)
        rot_mat = trans_mat[0:3, 0:3]
        inv_rot_mat = rot_mat.T
        vel_vec = np.array(
            [
                vehicle.get_velocity().x,
                vehicle.get_velocity().y,
                vehicle.get_velocity().z,
            ]
        ).reshape(3, 1)
        ego_velocity = (inv_rot_mat @ vel_vec).T[0]
        out_vel_state = VelocityReport()
        header = Header()
        header.stamp = ros_node.get_clock().now().to_msg()
        header.frame_id = 'base_link'
        out_vel_state.header = header
        out_vel_state.longitudinal_velocity = ego_velocity[0]
        out_vel_state.lateral_velocity = ego_velocity[1]
        out_vel_state.heading_rate = (
            vehicle.get_transform().transform_vector(vehicle.get_angular_velocity()).z
        )
        ros_node.pub_vel_state.publish(out_vel_state)


        out_steering_state = SteeringReport()
        out_ctrl_mode = ControlModeReport()
        out_gear_state = GearReport()
        out_actuation_status = ActuationStatusStamped()

        out_steering_state.stamp = out_vel_state.header.stamp
        out_steering_state.steering_tire_angle = -math.radians(
            vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
        )

        out_gear_state.stamp = out_vel_state.header.stamp
        out_gear_state.report = GearReport.DRIVE

        out_ctrl_mode.stamp = out_vel_state.header.stamp
        out_ctrl_mode.mode = ControlModeReport.AUTONOMOUS

        control = vehicle.get_control()
        out_actuation_status.header = header
        out_actuation_status.status.accel_status = control.throttle
        out_actuation_status.status.brake_status = control.brake
        out_actuation_status.status.steer_status = -control.steer

        ros_node.pub_actuation_status.publish(out_actuation_status)
        ros_node.pub_vel_state.publish(out_vel_state)
        ros_node.pub_steering_state.publish(out_steering_state)
        ros_node.pub_ctrl_mode.publish(out_ctrl_mode)
        ros_node.pub_gear_state.publish(out_gear_state)
        '''

        '''
        clock_msg = Clock()
        now = ros_node.get_clock().now()
        clock_msg.clock = now.to_msg()
        ros_node.clock_publisher.publish(clock_msg)
        '''

        # After you build out_pose_with_cov.pose.orientation as 'ros_quat'
        carla_yaw_deg = vehicle.get_transform().rotation.yaw
        carla_yaw = math.radians(carla_yaw_deg)
        ros_yaw = quat_to_yaw(out_pose_with_cov.pose.pose.orientation)

        diff = (ros_yaw - carla_yaw + math.pi) % (2*math.pi) - math.pi  # normalized to [-pi,pi]
        #print(f"carla_yaw (deg)={carla_yaw_deg:.2f}, ros_yaw(deg)={math.degrees(ros_yaw):.2f}, diff_deg={math.degrees(diff):.2f}")



    lidar1.listen(lambda data: lidar_callback(data, 'velodyne_top_changed', ros_node.lidar_top_queue))
    lidar2.listen(lambda data: lidar_callback(data, 'velodyne_right_changed', ros_node.lidar_right_queue))
    lidar3.listen(lambda data: lidar_callback(data, 'velodyne_left_changed', ros_node.lidar_left_queue))
    imu.listen(lambda data: imu_callback(data, ros_node.imu_queue))
    gnss.listen(lambda data: pose_callback(data, ros_node.pose_queue))

    def secs_to_time(sec_float):
        sec = int(sec_float)
        nsec = int((sec_float - sec) * 1e9)
        t = Time()
        t.sec = sec
        t.nanosec = nsec
        return t

    def quat_to_yaw(q):
        # q: geometry_msgs/Quaternion
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw
    
    


    # Tick simulation at 20Hz for N frames
    frame_count = 0
    try:
        while rclpy.ok() and frame_count < 1000000:
            world.tick()  # Sync mode: tick at each loop
            rclpy.spin_once(ros_node, timeout_sec=0)

            # inside your loop (replace your sim_time creation)
            # snapshot.timestamp is CARLA timestamp object
            snapshot = world.get_snapshot()           
            sim_seconds = snapshot.timestamp.elapsed_seconds
            sim_time_msg = secs_to_time(sim_seconds)

            sim_time_msg = ros_node.get_clock().now().to_msg()
            
            clock_msg = Clock()
            clock_msg.clock = sim_time_msg
            ros_node.clock_publisher.publish(clock_msg)

            '''            
            expected_keys = ["velodyne_top_changed", "velodyne_right_changed", "velodyne_left_changed", "imu", "pose"]
            if all(k in sensor_data for k in expected_keys):
                # Set header.stamp for each msg to sim_time
                for k, msg in sensor_data.items():
                    msg.header.stamp = sim_time_msg
                    if "velodyne_top_changed" in k:
                        ros_node.publisher1.publish(msg)
                    elif "velodyne_right_changed" in k:
                        ros_node.publisher2.publish(msg)
                    elif "velodyne_left_changed" in k:
                        ros_node.publisher3.publish(msg)
                    elif "imu" in k:
                        ros_node.imu_publisher.publish(msg)
                    elif "pose" in k:
                        ros_node.pose_publisher.publish(msg)   
            else:
                print(f"Missing sensors for tick: {set(expected_keys)-set(sensor_data.keys())}")
            '''
            try:
                top_msg = ros_node.lidar_top_queue.get(timeout=1.0)
                right_msg = ros_node.lidar_right_queue.get(timeout=1.0)
                left_msg = ros_node.lidar_left_queue.get(timeout=1.0)
                imu_msg = ros_node.imu_queue.get(timeout=1.0)
                pose_msg = ros_node.pose_queue.get(timeout=1.0)
            except Empty:
                print("Timed out waiting for sensor data")
                continue

            imu_msg.header.stamp = sim_time_msg
            ros_node.imu_publisher.publish(imu_msg)

            pose_msg.header.stamp = sim_time_msg
            ros_node.pose_publisher.publish(pose_msg)

            # Stamp and publish all with the same CARLA time
            for msg, pub, frame_id in [
                (top_msg, ros_node.publisher1, "velodyne_top_changed"),
                (right_msg, ros_node.publisher2, "velodyne_right_changed"),
                (left_msg, ros_node.publisher3, "velodyne_left_changed"),
            ]:
                msg.header.stamp = sim_time_msg
                msg.header.frame_id = frame_id
                pub.publish(msg)




            trans_mat = np.array(vehicle.get_transform().get_matrix()).reshape(4, 4)
            rot_mat = trans_mat[0:3, 0:3]
            inv_rot_mat = rot_mat.T
            vel_vec = np.array(
                [
                    vehicle.get_velocity().x,
                    vehicle.get_velocity().y,
                    vehicle.get_velocity().z,
                ]
            ).reshape(3, 1)
            ego_velocity = (inv_rot_mat @ vel_vec).T[0]
            out_vel_state = VelocityReport()
            header = Header()
            header.stamp = sim_time_msg
            header.frame_id = 'base_link'
            out_vel_state.header = header
            out_vel_state.longitudinal_velocity = ego_velocity[0]
            out_vel_state.lateral_velocity = ego_velocity[1]
            out_vel_state.heading_rate = (
                vehicle.get_transform().transform_vector(vehicle.get_angular_velocity()).z
            )
            ros_node.pub_vel_state.publish(out_vel_state)


            out_steering_state = SteeringReport()
            out_ctrl_mode = ControlModeReport()
            out_gear_state = GearReport()
            out_actuation_status = ActuationStatusStamped()

            out_steering_state.stamp = out_vel_state.header.stamp
            out_steering_state.steering_tire_angle = -math.radians(
                vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
            )

            out_gear_state.stamp = out_vel_state.header.stamp
            out_gear_state.report = GearReport.DRIVE

            out_ctrl_mode.stamp = out_vel_state.header.stamp
            out_ctrl_mode.mode = ControlModeReport.AUTONOMOUS

            control = vehicle.get_control()
            out_actuation_status.header = header
            out_actuation_status.status.accel_status = control.throttle
            out_actuation_status.status.brake_status = control.brake
            out_actuation_status.status.steer_status = -control.steer

            ros_node.pub_actuation_status.publish(out_actuation_status)
            ros_node.pub_vel_state.publish(out_vel_state)
            ros_node.pub_steering_state.publish(out_steering_state)
            ros_node.pub_ctrl_mode.publish(out_ctrl_mode)
            ros_node.pub_gear_state.publish(out_gear_state)

            print('iteration', frame_count)
            #sensor_data = {}
            #time.sleep(0.05)
            frame_count += 1
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user (Ctrl+C). Cleaning up and exiting...")
    finally:
        lidar1.stop()
        lidar1.destroy()
        lidar2.stop()
        lidar2.destroy()
        lidar3.stop()
        lidar3.destroy()
        imu.stop()
        imu.destroy()
        gnss.stop()
        gnss.destroy()
        
        if vehicle is not None:
            vehicle.destroy()

        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        rclpy.shutdown()

if __name__ == '__main__':
    # Start ROS2 node
    rclpy.init()
    main()
