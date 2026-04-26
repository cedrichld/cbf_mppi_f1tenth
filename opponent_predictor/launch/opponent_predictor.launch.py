from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    waypoint_path = LaunchConfiguration('waypoint_path')

    predictor = Node(
        package='opponent_predictor',
        executable='opponent_predictor_node',
        name='opponent_predictor',
        output='screen',
        parameters=[{
            'input_odom_topic': LaunchConfiguration('input_odom_topic'),
            'waypoint_path': waypoint_path,
            'waypoint_path_absolute': True,
            'frame_id': LaunchConfiguration('frame_id'),
            'pose_source': LaunchConfiguration('pose_source'),
            'tf_lookup_timeout': ParameterValue(
                LaunchConfiguration('tf_lookup_timeout'),
                value_type=float,
            ),
            'prediction_steps': ParameterValue(LaunchConfiguration('prediction_steps'), value_type=int),
            'prediction_dt': ParameterValue(LaunchConfiguration('prediction_dt'), value_type=float),
            'publish_rate_hz': ParameterValue(LaunchConfiguration('publish_rate_hz'), value_type=float),
            'speed_profile_scale': ParameterValue(LaunchConfiguration('speed_profile_scale'), value_type=float),
            'speed_profile_min_speed': ParameterValue(LaunchConfiguration('speed_profile_min_speed'), value_type=float),
            'speed_profile_max_speed': ParameterValue(LaunchConfiguration('speed_profile_max_speed'), value_type=float),
            'profile_speed_blend': ParameterValue(LaunchConfiguration('profile_speed_blend'), value_type=float),
            'out_of_sight_profile_speed_blend': ParameterValue(
                LaunchConfiguration('out_of_sight_profile_speed_blend'),
                value_type=float,
            ),
            'lateral_offset_alpha': ParameterValue(
                LaunchConfiguration('lateral_offset_alpha'),
                value_type=float,
            ),
            'lateral_offset_decay_time': ParameterValue(
                LaunchConfiguration('lateral_offset_decay_time'),
                value_type=float,
            ),
            'position_snap_alpha': ParameterValue(
                LaunchConfiguration('position_snap_alpha'),
                value_type=float,
            ),
            'max_progress_speed': ParameterValue(
                LaunchConfiguration('max_progress_speed'),
                value_type=float,
            ),
            'max_progress_accel': ParameterValue(
                LaunchConfiguration('max_progress_accel'),
                value_type=float,
            ),
            'kf_process_var_s': ParameterValue(
                LaunchConfiguration('kf_process_var_s'),
                value_type=float,
            ),
            'kf_process_var_v': ParameterValue(
                LaunchConfiguration('kf_process_var_v'),
                value_type=float,
            ),
            'kf_measurement_var_s': ParameterValue(
                LaunchConfiguration('kf_measurement_var_s'),
                value_type=float,
            ),
            'kf_measurement_var_v': ParameterValue(
                LaunchConfiguration('kf_measurement_var_v'),
                value_type=float,
            ),
            'stale_timeout': ParameterValue(LaunchConfiguration('stale_timeout'), value_type=float),
            'max_stale_prediction_time': ParameterValue(
                LaunchConfiguration('max_stale_prediction_time'),
                value_type=float,
            ),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'input_odom_topic',
            default_value='/opp_racecar/odom',
            description='Opponent odometry source. In sim this can be the ground-truth opponent odom.',
        ),
        DeclareLaunchArgument(
            'waypoint_path',
            default_value=PathJoinSubstitution([
                FindPackageShare('mppi_bringup'),
                'waypoints',
                'lev_testing',
                'lev_blocked.csv',
            ]),
            description='Raceline CSV used to project and predict opponent progress.',
        ),
        DeclareLaunchArgument('frame_id', default_value='map'),
        DeclareLaunchArgument(
            'pose_source',
            default_value='odom_pose',
            description='odom_pose trusts map-frame odom directly; tf forces TF transform when needed.',
        ),
        DeclareLaunchArgument('tf_lookup_timeout', default_value='0.02'),
        DeclareLaunchArgument('prediction_steps', default_value='5'),
        DeclareLaunchArgument('prediction_dt', default_value='0.1'),
        DeclareLaunchArgument('publish_rate_hz', default_value='20.0'),
        DeclareLaunchArgument('speed_profile_scale', default_value='1.0'),
        DeclareLaunchArgument('speed_profile_min_speed', default_value='0.0'),
        DeclareLaunchArgument('speed_profile_max_speed', default_value='20.0'),
        DeclareLaunchArgument(
            'profile_speed_blend',
            default_value='0.2',
            description='0: predict with filtered opponent speed; 1: predict with raceline speed.',
        ),
        DeclareLaunchArgument(
            'out_of_sight_profile_speed_blend',
            default_value='1.0',
            description='Blend used after opponent odom goes stale.',
        ),
        DeclareLaunchArgument(
            'lateral_offset_alpha',
            default_value='0.35',
            description='Filter gain for measured lateral offset from the raceline.',
        ),
        DeclareLaunchArgument(
            'lateral_offset_decay_time',
            default_value='1.0',
            description='Seconds for predicted lateral offset to decay back toward the raceline.',
        ),
        DeclareLaunchArgument(
            'position_snap_alpha',
            default_value='1.0',
            description='1 anchors fresh opponent position to odom projection; lower values smooth position.',
        ),
        DeclareLaunchArgument('stale_timeout', default_value='0.5'),
        DeclareLaunchArgument('max_stale_prediction_time', default_value='2.0'),
        DeclareLaunchArgument('max_progress_speed', default_value='12.0'),
        DeclareLaunchArgument('max_progress_accel', default_value='8.0'),
        DeclareLaunchArgument('kf_process_var_s', default_value='0.05'),
        DeclareLaunchArgument('kf_process_var_v', default_value='0.5'),
        DeclareLaunchArgument('kf_measurement_var_s', default_value='0.15'),
        DeclareLaunchArgument('kf_measurement_var_v', default_value='0.6'),
        predictor,
    ])
