import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float64MultiArray, Header
from tf2_ros import TransformBroadcaster

from dobot_msgs.msg import DobotAlarmCodes, GripperStatus
from dobot_msgs.srv import (
    ExecuteHomingProcedure,
    GripperControl,
    SuctionCupControl,
    EvaluatePTPTrajectory,
)
from dobot_msgs.action import PointToPoint


def deg2rad(d):
    return d * math.pi / 180.0


def rad2deg(r):
    return r * 180.0 / math.pi


def forward_kinematics(j1_deg, j2_deg, j3_deg, j4_deg):
    L1 = 135.0
    L2 = 147.0
    L3 = 60.0
    j1 = deg2rad(j1_deg)
    j2 = deg2rad(j2_deg)
    j3 = deg2rad(j3_deg)

    r = L2 * math.cos(j2) + L3 * math.cos(j3)
    x = r * math.cos(j1)
    y = r * math.sin(j1)
    z = L1 + L2 * math.sin(j2) + L3 * math.sin(j3)
    return x, y, z, j4_deg


def inverse_kinematics(x, y, z, r):
    L1 = 135.0
    L2 = 147.0
    L3 = 60.0

    j1 = math.atan2(y, x)
    rr = math.sqrt(x * x + y * y)
    zz = z - L1

    dist = math.sqrt(rr * rr + zz * zz)
    dist = min(dist, L2 + L3 - 0.01)

    cos_j3_rel = (dist * dist - L2 * L2 - L3 * L3) / (2 * L2 * L3)
    cos_j3_rel = max(-1.0, min(1.0, cos_j3_rel))
    j3_rel = math.acos(cos_j3_rel)

    alpha = math.atan2(zz, rr)
    beta = math.atan2(L3 * math.sin(j3_rel), L2 + L3 * math.cos(j3_rel))
    j2 = alpha + beta
    j3 = j2 - j3_rel

    return rad2deg(j1), rad2deg(j2), rad2deg(j3), r


class DobotDummyNode(Node):
    def __init__(self):
        super().__init__('dobot_dummy_node')
        self.get_logger().info('Dobot Dummy Simulator starting...')

        self.cb_group = ReentrantCallbackGroup()

        self.joint_angles_deg = [0.0, 45.0, 45.0, 0.0]
        x, y, z, r = forward_kinematics(*self.joint_angles_deg)
        self.tcp_position = [x, y, z, r]
        self.gripper_state = 'open'
        self.gripper_width = 0.01
        self.suction_enabled = False
        self.is_moving = False
        self.lock = threading.Lock()

        self.joint_names = [
            'magician_joint_1',
            'magician_joint_2',
            'magician_joint_3',
            'magician_joint_4',
            'magician_joint_prismatic_l',
        ]

        self.pub_joint_states = self.create_publisher(JointState, 'joint_states', 10)
        self.pub_dobot_joint_states = self.create_publisher(JointState, 'dobot_joint_states', 10)
        self.pub_tcp = self.create_publisher(PoseStamped, 'dobot_TCP', 10)
        self.pub_pose_raw = self.create_publisher(Float64MultiArray, 'dobot_pose_raw', 10)
        self.pub_alarms = self.create_publisher(DobotAlarmCodes, 'dobot_alarms', 10)
        self.pub_gripper_status = self.create_publisher(GripperStatus, 'gripper_status_rviz', 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_timer(0.05, self.publish_state)
        self.create_timer(0.1, self.publish_alarms)

        self.homing_srv = self.create_service(
            ExecuteHomingProcedure,
            'dobot_homing_service',
            self.homing_callback,
            callback_group=self.cb_group,
        )

        self.gripper_srv = self.create_service(
            GripperControl,
            'dobot_gripper_service',
            self.gripper_callback,
            callback_group=self.cb_group,
        )

        self.suction_srv = self.create_service(
            SuctionCupControl,
            'dobot_suction_cup_service',
            self.suction_callback,
            callback_group=self.cb_group,
        )

        self.validation_srv = self.create_service(
            EvaluatePTPTrajectory,
            'dobot_PTP_validation_service',
            self.validation_callback,
            callback_group=self.cb_group,
        )

        self.ptp_action_server = ActionServer(
            self,
            PointToPoint,
            'PTP_action',
            execute_callback=self.ptp_execute_callback,
            goal_callback=self.ptp_goal_callback,
            cancel_callback=self.ptp_cancel_callback,
            callback_group=self.cb_group,
        )

        self.get_logger().info('Dobot Dummy Simulator ready! (no hardware needed)')
        self.get_logger().info(f'Initial TCP position: x={x:.1f}, y={y:.1f}, z={z:.1f}, r={r:.1f}')

    def publish_state(self):
        now = self.get_clock().now().to_msg()

        with self.lock:
            j1, j2, j3, j4 = self.joint_angles_deg
            tcp = list(self.tcp_position)
            gw = self.gripper_width

        j1_rad = deg2rad(j1)
        j2_rad = deg2rad(j2)
        j3_rad = deg2rad(j3)
        j4_rad = deg2rad(j4)

        js_rviz = JointState()
        js_rviz.header.stamp = now
        js_rviz.name = self.joint_names
        js_rviz.position = [j1_rad, j2_rad, j3_rad - j2_rad, j4_rad, gw]
        self.pub_joint_states.publish(js_rviz)

        js_dobot = JointState()
        js_dobot.header.stamp = now
        js_dobot.name = self.joint_names
        js_dobot.position = [j1_rad, j2_rad, j3_rad, j4_rad, gw]
        self.pub_dobot_joint_states.publish(js_dobot)

        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = 'magician_base_link'
        pose.pose.position.x = tcp[0] / 1000.0
        pose.pose.position.y = tcp[1] / 1000.0
        pose.pose.position.z = tcp[2] / 1000.0
        yaw = deg2rad(tcp[3])
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub_tcp.publish(pose)

        raw = Float64MultiArray()
        raw.data = [tcp[0] / 1000.0, tcp[1] / 1000.0, tcp[2] / 1000.0, tcp[3]]
        self.pub_pose_raw.publish(raw)

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'magician_base_link'
        t.child_frame_id = 'TCP'
        t.transform.translation.x = tcp[0] / 1000.0
        t.transform.translation.y = tcp[1] / 1000.0
        t.transform.translation.z = tcp[2] / 1000.0
        t.transform.rotation = pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def publish_alarms(self):
        msg = DobotAlarmCodes()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.alarms_list = []
        self.pub_alarms.publish(msg)

    def homing_callback(self, request, response):
        self.get_logger().info('Homing procedure started (dummy)...')
        time.sleep(1.0)
        with self.lock:
            self.joint_angles_deg = [0.0, 45.0, 45.0, 0.0]
            self.tcp_position = list(forward_kinematics(*self.joint_angles_deg))
        self.get_logger().info(f'Homing complete. TCP: {self.tcp_position}')
        response.success = True
        response.instruction = 'Homing completed successfully (dummy mode)'
        return response

    def gripper_callback(self, request, response):
        with self.lock:
            self.gripper_state = request.gripper_state
            if request.gripper_state == 'open':
                self.gripper_width = 0.01
            else:
                self.gripper_width = 0.0

        status_msg = GripperStatus()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        status_msg.status = request.gripper_state
        self.pub_gripper_status.publish(status_msg)

        self.get_logger().info(f'Gripper: {request.gripper_state}')
        response.success = True
        response.message = f'Gripper {request.gripper_state} (dummy mode)'
        return response

    def suction_callback(self, request, response):
        with self.lock:
            self.suction_enabled = request.enable_suction
        state = 'ON' if request.enable_suction else 'OFF'
        self.get_logger().info(f'Suction cup: {state}')
        response.success = True
        response.message = f'Suction cup {state} (dummy mode)'
        return response

    def validation_callback(self, request, response):
        target = request.target
        x_range = (0.0, 320.0)
        y_range = (-320.0, 320.0)
        z_range = (-50.0, 200.0)

        if request.motion_type in [1, 2]:
            r = math.sqrt(target[0] ** 2 + target[1] ** 2)
            if r > 320.0 or r < 100.0:
                response.is_valid = False
                response.message = 'Target out of reach'
                return response
            if target[2] < z_range[0] or target[2] > z_range[1]:
                response.is_valid = False
                response.message = 'Z coordinate out of range'
                return response

        response.is_valid = True
        response.message = 'Trajectory is valid (dummy mode)'
        return response

    def ptp_goal_callback(self, goal_request):
        if goal_request.motion_type not in [1, 2, 4, 5]:
            self.get_logger().warn(f'Invalid motion type: {goal_request.motion_type}')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def ptp_cancel_callback(self, goal_handle):
        self.get_logger().info('Motion cancelled')
        with self.lock:
            self.is_moving = False
        return CancelResponse.ACCEPT

    def ptp_execute_callback(self, goal_handle):
        self.get_logger().info('Executing PTP motion (dummy)...')

        target = list(goal_handle.request.target_pose)
        motion_type = goal_handle.request.motion_type
        vel_ratio = goal_handle.request.velocity_ratio
        acc_ratio = goal_handle.request.acceleration_ratio

        vel_ratio = max(0.1, min(1.0, vel_ratio))

        if motion_type in [4, 5]:
            target_joints = target
            target_tcp = list(forward_kinematics(*target_joints))
        else:
            target_tcp = target
            target_joints = list(inverse_kinematics(*target_tcp))

        with self.lock:
            start_joints = list(self.joint_angles_deg)
            self.is_moving = True

        steps = int(40 / vel_ratio)
        steps = max(10, steps)

        feedback_msg = PointToPoint.Feedback()

        for i in range(1, steps + 1):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                with self.lock:
                    self.is_moving = False
                result = PointToPoint.Result()
                result.achieved_pose = self.tcp_position
                return result

            t = i / float(steps)
            t_smooth = t * t * (3.0 - 2.0 * t)

            current_joints = [
                start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                for j in range(4)
            ]

            current_tcp = list(forward_kinematics(*current_joints))

            with self.lock:
                self.joint_angles_deg = current_joints
                self.tcp_position = current_tcp

            if motion_type in [4, 5]:
                feedback_msg.current_pose = current_joints
            else:
                feedback_msg.current_pose = current_tcp

            goal_handle.publish_feedback(feedback_msg)
            time.sleep(0.05)

        with self.lock:
            self.joint_angles_deg = target_joints
            self.tcp_position = target_tcp
            self.is_moving = False

        goal_handle.succeed()
        result = PointToPoint.Result()
        result.achieved_pose = target_tcp
        self.get_logger().info(f'Motion complete. TCP: [{target_tcp[0]:.1f}, {target_tcp[1]:.1f}, {target_tcp[2]:.1f}, {target_tcp[3]:.1f}]')
        return result


def main(args=None):
    rclpy.init(args=args)
    node = DobotDummyNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
