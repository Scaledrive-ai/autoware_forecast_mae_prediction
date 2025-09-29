import rclpy
from rclpy.node import Node
from rclpy.time import Time

from tf_transformations import euler_from_quaternion
import math
import os
import sys
import numpy as np
import joblib
import uuid
import torch
import time
from collections import deque
import torch.nn.functional as F  # for softmax

from autoware_perception_msgs.msg import TrackedObjects, Shape
from nav_msgs.msg import Odometry
from lanelet2.projection import LocalCartesianProjector
from lanelet2.io import load, Origin
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseWithCovariance

from .model_lane_input import get_centerline
from .model_agent_input import build_model_inputs
from .util import uuid_to_str, make_ros_uuid
from .forecast_mae.model.multiagent.trainer_forecast_ma import Trainer as Model

from autoware_perception_msgs.msg import PredictedObjects, PredictedObject, PredictedObjectKinematics, PredictedPath, ObjectClassification
from unique_identifier_msgs.msg import UUID
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Duration


class MotionPredictionNode(Node):
    def __init__(self):
        super().__init__('forecast_mae_motion_prediction_node')
        self.get_logger().info("loading forecast-mae motion prediction model...")
        
        self.declare_parameter('max_num_agents', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('marker_array_vis_output', rclpy.Parameter.Type.STRING)
        self.declare_parameter('predicted_obj_output', rclpy.Parameter.Type.STRING)
        self.declare_parameter('lanelet_map_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('model_ckpt_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('map_load_interval', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('prediction_frequency', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('agent_reset_time_gap', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('ex_agent_hold_period', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('time_history', rclpy.Parameter.Type.INTEGER)
        
        self.max_num_agents = self.get_parameter('max_num_agents').get_parameter_value().integer_value
        self.agent_reset_time_gap = self.get_parameter('agent_reset_time_gap').get_parameter_value().double_value
        self.ex_agent_hold_period = self.get_parameter('ex_agent_hold_period').get_parameter_value().double_value
        self.time_history = self.get_parameter('time_history').get_parameter_value().integer_value * 10
        self.ego_pose = None  # Store the latest ego pose 
        self.agent_buffers = {}
        self.start_time = None
        self.first_map_load = True
        self.last_marker_ids = []


        #subscribers
        self.subscription = self.create_subscription(
            TrackedObjects,
            '/perception/object_recognition/tracking/objects',
            self.objects_callback,
            10
        )
        self.ego_pose_sub = self.create_subscription(
            Odometry,
            '/localization/kinematic_state',
            self.ego_pose_callback,
            1
        )
        
        #publishers
        marker_pub_topic = self.get_parameter('marker_array_vis_output').get_parameter_value().string_value
        obj_pub_topic = self.get_parameter('predicted_obj_output').get_parameter_value().string_value
        self.pub = self.create_publisher(MarkerArray, marker_pub_topic, 10)
        self.pred_pub = self.create_publisher(PredictedObjects, obj_pub_topic, 10)

	# setting up timer to call prediction and map update
        PREDICTION_HZ = self.get_parameter('prediction_frequency').get_parameter_value().double_value  # how often to run inference
        MAP_LOAD_INTERVAL = self.get_parameter('map_load_interval').get_parameter_value().double_value
        self.prediction_timer = self.create_timer(1.0 / PREDICTION_HZ, self.prediction_callback)
        self.map_loader = self.create_timer(MAP_LOAD_INTERVAL, self.map_load_callback)

        # Load lanelet map and setup
        map_path = self.get_parameter('lanelet_map_path').get_parameter_value().string_value
        map_path = os.path.expanduser(map_path)
        assert os.path.exists(map_path), f"Map file not found at {map_path}"
        proj = LocalCartesianProjector(Origin(0, 0, 0))
        self.lanelet_map = load(map_path, proj)
        self.lane_centerlines, self.lane_attrs, self.is_intersections = None, None, None
        
        # load prediction model checkpoint
        ckpt_path = self.get_parameter('model_ckpt_path').get_parameter_value().string_value
        assert os.path.exists(ckpt_path), "chkpt files does not exist, update path to checkpoint"
        model = Model.load_from_checkpoint(ckpt_path, pretrained_weights=ckpt_path)
        self.model = model.eval().cuda()



    def get_buffer_template(self, last_timestamp, label, last_orientation_quat, bb_dimensions, max_hist=50):
        buffer = {
            "pos": deque(maxlen=max_hist),
            "heading": deque(maxlen=max_hist),
            "velocity": deque(maxlen=max_hist),
            "last_timestamp": last_timestamp,
            "label": label,
            "last_orientation_quat": last_orientation_quat,
            "bb_dimensions": bb_dimensions
        }
        return buffer

    def update_agent_history(self, agent_id, pos, heading, velocity, last_timestamp, label, last_orientation_quat, bb_dimensions):
        """Update or create rolling history for an agent."""
        if agent_id not in self.agent_buffers and len(self.agent_buffers.keys()) < self.max_num_agents:
            self.get_logger().info(f'creating buffer for new agent, id: {agent_id}')
            self.agent_buffers[agent_id] = self.get_buffer_template(last_timestamp, label, last_orientation_quat, bb_dimensions)

        self.agent_buffers[agent_id]["pos"].append(pos)
        self.agent_buffers[agent_id]["heading"].append(heading)
        self.agent_buffers[agent_id]["velocity"].append(velocity)
        self.agent_buffers[agent_id]["last_timestamp"] = last_timestamp
        self.agent_buffers[agent_id]["label"] = label
        self.agent_buffers[agent_id]["last_orientation_quat"] = last_orientation_quat
        self.agent_buffers[agent_id]["bb_dimensions"] = bb_dimensions

    def ego_pose_callback(self, msg):
        self.ego_pose = msg.pose.pose

        if self.first_map_load:
            self.get_logger().info('loading map...')

            ego_pos_x = self.ego_pose.position.x
            ego_pos_y = self.ego_pose.position.y
            self.lane_centerlines, self.lane_attrs, self.is_intersections = get_centerline(self.lanelet_map, ego_pos_x=ego_pos_x, ego_pos_y=ego_pos_y)
            self.first_map_load = False

    def objects_callback(self, msg):
        if self.ego_pose is None:
            self.get_logger().info("Waiting for ego pose...")
            return

        for obj_id, obj in enumerate(msg.objects):  # iterate through detected objects
            if obj.classification[0].label == 1 or obj.classification[0].label == 3 or obj.classification[0].label == 7:
                agent_id = obj.object_id
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                agent_id = uuid_to_str(obj.object_id)

                #position
                pos = obj.kinematics.pose_with_covariance.pose.position
                # Transform position from Autoware to CARLA coordinate system
                pos_x = pos.x  
                pos_y = pos.y  
                
                #orientation
                ori = obj.kinematics.pose_with_covariance.pose.orientation
                quat = (ori.x,ori.y,ori.z,ori.w)
                _, _, yaw = euler_from_quaternion(quat)
                yaw *= -1
                
                #velocity
                # Transform velocity from Autoware to CARLA coordinate system
                vx_o = obj.kinematics.twist_with_covariance.twist.linear.x
                vy_o = obj.kinematics.twist_with_covariance.twist.linear.y
                # 3) rotate into map frame
                vx_map = math.cos(yaw) * vx_o - math.sin(yaw) * vy_o
                vy_map = math.sin(yaw) * vx_o + math.cos(yaw) * vy_o
                vel_norm = np.linalg.norm([vx_map, vy_map])

                bb_dimensions=(obj.shape.dimensions.x, obj.shape.dimensions.y, obj.shape.dimensions.z)

                if agent_id in self.agent_buffers.keys():
                    last_time = self.agent_buffers[agent_id]["last_timestamp"]
                    delta = timestamp - last_time
                    if abs(delta) > self.agent_reset_time_gap:  # Tolerance = 1.5s
                        self.get_logger().info(f"Resetting buffer for {agent_id} due to gap: {delta:.3f}s")
                        label = obj.classification[0].label
                        self.agent_buffers[agent_id] = self.get_buffer_template(timestamp, label, quat, bb_dimensions)

                self.update_agent_history(
                    agent_id,
                    pos=np.array([pos_x, pos_y]),
                    heading=yaw,
                    velocity=vel_norm,
                    last_timestamp=timestamp,
                    label=obj.classification[0].label,
                    last_orientation_quat=quat,
                    bb_dimensions=bb_dimensions
                )



    def map_load_callback(self):
        if self.ego_pose:
            ego_pos_x = self.ego_pose.position.x
            ego_pos_y = self.ego_pose.position.y
            self.lane_centerlines, self.lane_attrs, self.is_intersections = get_centerline(self.lanelet_map, ego_pos_x=ego_pos_x, ego_pos_y=ego_pos_y)
            self.get_logger().info('loading map...')


    def prediction_callback(self):
        now = self.get_clock().now().seconds_nanoseconds()
        sec, nsec = now
        timestamp = sec + nsec * 1e-9

        padded_buffers = {}

        for agent_id, buf in list(self.agent_buffers.items()):
            last_time = buf["last_timestamp"]
            delta = timestamp - last_time
            if abs(delta) > self.ex_agent_hold_period:
                del self.agent_buffers[agent_id]
                self.get_logger().info(f"Deleting agent with id {agent_id}, stale {delta:.2f}s")
                continue

            n = len(buf["pos"])
            if n < self.time_history:
                # too little data → skip
                continue

            if n < 50:
                pad_len = 50 - n
                first_pos = buf["pos"][0]
                first_heading = buf["heading"][0]
                # stationary padding
                padded_pos = [first_pos] * pad_len + list(buf["pos"])
                padded_heading = [first_heading] * pad_len + list(buf["heading"])
                padded_vel = [0.0] * pad_len + list(buf["velocity"])
            else:
                padded_pos = list(buf["pos"])
                padded_heading = list(buf["heading"])
                padded_vel = list(buf["velocity"])

            padded_buffers[agent_id] = {
                "pos": padded_pos,
                "heading": padded_heading,
                "velocity": padded_vel,
                "last_timestamp": buf["last_timestamp"],
                "label": buf["label"],
                "last_orientation_quat": buf["last_orientation_quat"],
                "bb_dimensions": buf["bb_dimensions"]
            }

        success, agent_state_data, lanelet_data = build_model_inputs(padded_buffers, self.lane_centerlines, self.lane_attrs, self.is_intersections)
        

        
        if success:
            (x, 
            x_positions,
            x_ctrs, 
            x_attr, 
            x_heading, 
            x_velocity, 
            x_velocity_diff, 
            padding_mask, 
            scored_agents_mask, 
            final_actor_ids,
            cur_pos, 
            cur_heading,
            cur_pos_all,
            cur_heading_all) = agent_state_data

            (lane_positions_torch, 
            lane_attrs_updated, 
            is_intersections_updated, 
            lanes_ctr, 
            lanes_angle, 
            lane_padding_mask) = lanelet_data

            prediction_data = {
                "x": x[:, :, :50],
                "x_attr": x_attr,
                "x_positions": x_positions,
                "x_centers": x_ctrs,
                "x_angles": x_heading,
                "x_velocity": x_velocity,
                "x_velocity_diff": x_velocity_diff,
                "x_padding_mask": padding_mask,
                "x_scored": scored_agents_mask,
                "lane_positions": lane_positions_torch,
                "lane_centers": lanes_ctr,
                "lane_angles": lanes_angle,
                "lane_attr": lane_attrs_updated,
                "lane_padding_mask": lane_padding_mask,
                "is_intersections": is_intersections_updated,
                "av_index": torch.zeros(x.shape[0]), 
                "origin": cur_pos, # current global position of ego vehicle 
                "theta": cur_heading, # current ego vehicle heading
                "scenario_id": [f'test_scenario_{i}' for i in range(x.shape[0])],
                "track_id": final_actor_ids,
                "city": "Dallas",
            }

            prediction_data["x_key_padding_mask"] = prediction_data["x_padding_mask"].all(-1)
            prediction_data["lane_key_padding_mask"] = prediction_data["lane_padding_mask"].all(-1)
            prediction_data["num_actors"] = (~prediction_data["x_key_padding_mask"]).sum(-1)
            prediction_data["num_lanes"] = (~prediction_data["lane_key_padding_mask"]).sum(-1)


            for k in prediction_data.keys():
                if torch.is_tensor(prediction_data[k]): 
                    prediction_data[k] = prediction_data[k].cuda()
            with torch.no_grad():
                out = self.model.net(prediction_data)
    
            trajectory = out["y_hat"] # [num_scenario, 6, 53, 60, 2]
            probability = out["pi"]
            normalized_probability = False


            scenario_ids_list = prediction_data["scenario_id"]
            batch = len(scenario_ids_list)

            theta = prediction_data["theta"].double()
            rotate_mat = torch.stack(
                [
                    torch.cos(theta),
                    torch.sin(theta),
                    -torch.sin(theta),
                    torch.cos(theta),
                ],
                dim=1,
            ).view(batch, 1, 1, 2, 2)

            # Step 2: Expand rotation matrix to match (B, M, N, 60, 2)
            rotate_mat_exp = rotate_mat.expand(-1, trajectory.size(1), trajectory.size(2), -1, -1)
            
            # Step 3: Use per-agent origins (origin_all)
            origin_all = cur_pos_all.double()  # (B, N, 2)
            origin_all_exp = origin_all[:, None, :, None, :].cpu()  # (B, 1, N, 1, 2)

            # Step 4: Rotate and translate
            trajectory_rotated = torch.matmul(
                trajectory.double(), rotate_mat_exp
            ).cpu()  # (B, M, N, 60, 2)

            global_trajectory = (trajectory_rotated + origin_all_exp).cpu().numpy() # (num_scenarios, 6, 53, 60, 2)

            # Delete only my old markers
            delete_array = MarkerArray()
            for old_id in self.last_marker_ids:
                m = Marker()
                m.action = Marker.DELETE
                m.ns = old_id["ns"]
                m.id = old_id["id"]
                delete_array.markers.append(m)
            self.pub.publish(delete_array)

            pred_msg = PredictedObjects()
            pred_msg.header.stamp = self.get_clock().now().to_msg()
            pred_msg.header.frame_id = "map"

            marker_array = MarkerArray()
            self.last_marker_ids = []
            marker_id = 0
            num_scenarios, num_modes, num_agents, horizon, _ = global_trajectory.shape
            agent_id_list = list(padded_buffers.keys())
            
            # Softmax probabilities across modes
            prob_softmax = F.softmax(probability, dim=1).cpu().numpy()  # shape (num_scenarios, 6)

            for scenario_idx in range(num_scenarios):
                
                last_seen = padded_buffers[agent_id_list[scenario_idx]]["last_timestamp"]
                if timestamp - last_seen > 1.0:
                    continue

                obj = PredictedObject()
                # Assign object_id from your buffer
                focal_agent_id = list(padded_buffers.keys())[scenario_idx]
                #obj.object_id = UUID(uuid=focal_agent_id.encode("utf-8")[:16]) 
                obj.object_id = make_ros_uuid(focal_agent_id)
                obj.existence_probability = 1.0

                # Classification (use stored label from buffer)
                classification = ObjectClassification()
                classification.label = padded_buffers[focal_agent_id]["label"]
                classification.probability = 1.0  # tracking already determined label
                obj.classification.append(classification)

                # Kinematics + predicted paths
                kinematics = PredictedObjectKinematics()
                kinematics.predicted_paths = []

                shape = Shape()
                shape.type = Shape.BOUNDING_BOX
                shape.dimensions.x = padded_buffers[agent_id_list[scenario_idx]]["bb_dimensions"][0]
                shape.dimensions.y = padded_buffers[agent_id_list[scenario_idx]]["bb_dimensions"][1]
                shape.dimensions.z = padded_buffers[agent_id_list[scenario_idx]]["bb_dimensions"][2]
                obj.shape = shape

                # initial pose
                last_pose = Pose()
                last_buffer_pos = padded_buffers[agent_id_list[scenario_idx]]["pos"][-1]
                last_buffer_ori = padded_buffers[agent_id_list[scenario_idx]]["last_orientation_quat"]
                last_pose.position.x = float(last_buffer_pos[0])
                last_pose.position.y = float(last_buffer_pos[1])
                last_pose.position.z = 0.0
                last_pose.orientation.x = last_buffer_ori[0]
                last_pose.orientation.y = last_buffer_ori[1]
                last_pose.orientation.z = last_buffer_ori[2]
                last_pose.orientation.w = last_buffer_ori[3]
                kinematics.initial_pose_with_covariance.pose = last_pose

                for mode_idx in range(num_modes):
                    coords = global_trajectory[scenario_idx, mode_idx, 0]  # agent index 0
                    # coords shape = (60, 2)

                    path_msg = PredictedPath()
                    path_msg.confidence = float(prob_softmax[scenario_idx, mode_idx])

                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = f"scenario_{scenario_idx}"
                    marker.id = marker_id
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD

                    # Add trajectory points
                    for t in range(horizon):
                        pt = Point()
                        pt.x = float(coords[t, 0])
                        pt.y = float(coords[t, 1])
                        pt.z = 0.0
                        marker.points.append(pt)
                        pose = Pose()
                        pose.position.x = float(coords[t, 0])
                        pose.position.y = float(coords[t, 1])
                        pose.position.z = 0.0
                        # Orientation optional → leave as 0 quaternion
                        path_msg.path.append(pose)

                    # Time step between trajectory points
                    dt = 0.1  # 10 Hz predictions
                    path_msg.time_step = Duration(sec=int(dt), nanosec=int((dt - int(dt)) * 1e9))
                    kinematics.predicted_paths.append(path_msg)

                    # Style
                    marker.scale.x = 0.2  # line width
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 1.0

                    marker_array.markers.append(marker)
                    self.last_marker_ids.append({"ns": marker.ns, "id": marker.id})
                    marker_id += 1

                obj.kinematics = kinematics
                pred_msg.objects.append(obj)


            self.pub.publish(marker_array)
            self.pred_pub.publish(pred_msg)

       

def main(args=None):
    rclpy.init(args=args)
    node = MotionPredictionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
