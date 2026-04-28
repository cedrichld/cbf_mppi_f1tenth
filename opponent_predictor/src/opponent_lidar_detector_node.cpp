#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace
{

constexpr double kEps = 1e-9;

double yawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(
    2.0 * (q.w * q.z + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.w = std::cos(0.5 * yaw);
  q.z = std::sin(0.5 * yaw);
  q.x = 0.0;
  q.y = 0.0;
  return q;
}

std::vector<std::string> splitSemicolonLine(const std::string & line)
{
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ';')) {
    fields.push_back(field);
  }
  return fields;
}

struct Waypoint
{
  double s = 0.0;
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
};

struct TrackProjection
{
  double s = 0.0;
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double lateral_error = 0.0;
  double distance = std::numeric_limits<double>::infinity();
};

struct ScanPoint
{
  double map_x = 0.0;
  double map_y = 0.0;
  double base_x = 0.0;
  double base_y = 0.0;
};

struct ClusterCandidate
{
  std::vector<int> indices;
  double centroid_x = 0.0;
  double centroid_y = 0.0;
  double centroid_base_x = 0.0;
  double centroid_base_y = 0.0;
  double center_x = 0.0;
  double center_y = 0.0;
  double center_s = 0.0;
  double center_yaw = 0.0;
  double projection_distance = std::numeric_limits<double>::infinity();
  double range = 0.0;
  double extent_t = 0.0;
  double extent_n = 0.0;
  double correction_norm = 0.0;
  double score = std::numeric_limits<double>::infinity();
};

}  // namespace

class OpponentLidarDetectorNode : public rclcpp::Node
{
public:
  OpponentLidarDetectorNode()
  : Node("opponent_lidar_detector_node")
  {
    scan_topic_ = declare_parameter<std::string>("scan_topic", "/scan");
    ego_odom_topic_ = declare_parameter<std::string>("ego_odom_topic", "/ego_racecar/odom");
    map_topic_ = declare_parameter<std::string>("map_topic", "/map");
    detected_odom_topic_ = declare_parameter<std::string>("detected_odom_topic", "/opponent/detection_odom");
    detector_marker_topic_ =
      declare_parameter<std::string>("detector_marker_topic", "/opponent/detection_markers");
    detector_debug_topic_ =
      declare_parameter<std::string>("detector_debug_topic", "/opponent/detection_debug");

    waypoint_path_ = declare_parameter<std::string>("waypoint_path", "waypoints/lev_testing/lev_blocked.csv");
    waypoint_path_absolute_ = declare_parameter<bool>("waypoint_path_absolute", false);
    frame_id_ = declare_parameter<std::string>("frame_id", "map");
    base_frame_id_ = declare_parameter<std::string>("base_frame_id", "ego_racecar/base_link");

    refreshLiveParams();
    loadWaypoints(resolveWaypointPath(waypoint_path_, waypoint_path_absolute_));

    const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(1);
    const auto debug_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();

    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic_, sensor_qos,
      std::bind(&OpponentLidarDetectorNode::scanCallback, this, std::placeholders::_1));
    ego_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      ego_odom_topic_, sensor_qos,
      std::bind(&OpponentLidarDetectorNode::egoOdomCallback, this, std::placeholders::_1));
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      map_topic_, rclcpp::QoS(1).transient_local().reliable(),
      std::bind(&OpponentLidarDetectorNode::mapCallback, this, std::placeholders::_1));

    detection_pub_ = create_publisher<nav_msgs::msg::Odometry>(detected_odom_topic_, sensor_qos);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(detector_marker_topic_, debug_qos);
    debug_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(detector_debug_topic_, debug_qos);

    RCLCPP_INFO(
      get_logger(),
      "Opponent LiDAR detector ready: scan=%s ego=%s map=%s waypoints=%zu",
      scan_topic_.c_str(), ego_odom_topic_.c_str(), map_topic_.c_str(), waypoints_.size());
  }

private:
  void refreshLiveParams()
  {
    laser_x_offset_ = declareOrGetDouble("laser_x_offset", 0.0);
    laser_y_offset_ = declareOrGetDouble("laser_y_offset", 0.0);
    laser_yaw_offset_ = declareOrGetDouble("laser_yaw_offset", 0.0);
    opponent_length_ = std::max(0.05, declareOrGetDouble("opponent_length", 0.58));
    opponent_width_ = std::max(0.05, declareOrGetDouble("opponent_width", 0.31));
    max_center_correction_ = std::max(0.0, declareOrGetDouble("max_center_correction", 0.35));

    min_range_ = std::max(0.0, declareOrGetDouble("min_range", 0.15));
    max_range_ = std::max(min_range_, declareOrGetDouble("max_range", 8.0));
    min_detection_range_ = std::max(0.0, declareOrGetDouble("min_detection_range", 0.35));
    max_detection_range_ = std::max(min_detection_range_, declareOrGetDouble("max_detection_range", 6.0));
    front_fov_only_ = declareOrGetBool("front_fov_only", false);
    min_base_x_ = declareOrGetDouble("min_base_x", -0.5);

    require_static_map_ = declareOrGetBool("require_static_map", true);
    occupied_threshold_ = std::clamp(static_cast<int>(declareOrGetInt("occupied_threshold", 50)), 0, 100);
    wall_inflation_radius_ = std::max(0.0, declareOrGetDouble("wall_inflation_radius", 0.18));
    treat_unknown_as_static_ = declareOrGetBool("treat_unknown_as_static", true);

    cluster_tolerance_ = std::max(0.01, declareOrGetDouble("cluster_tolerance", 0.18));
    min_cluster_points_ = std::max(1, static_cast<int>(declareOrGetInt("min_cluster_points", 4)));
    max_cluster_points_ = std::max(min_cluster_points_, static_cast<int>(declareOrGetInt("max_cluster_points", 80)));
    min_cluster_extent_ = std::max(0.0, declareOrGetDouble("min_cluster_extent", 0.08));
    max_cluster_extent_ = std::max(min_cluster_extent_, declareOrGetDouble("max_cluster_extent", 0.9));

    max_raceline_projection_dist_ = std::max(0.0, declareOrGetDouble("max_raceline_projection_dist", 1.2));
    max_candidate_jump_ = std::max(0.0, declareOrGetDouble("max_candidate_jump", 1.5));
    continuity_gate_timeout_ = std::max(0.0, declareOrGetDouble("continuity_gate_timeout", 0.4));
    continuity_weight_ = std::max(0.0, declareOrGetDouble("continuity_weight", 2.0));
    projection_weight_ = std::max(0.0, declareOrGetDouble("projection_weight", 1.0));
    range_weight_ = std::max(0.0, declareOrGetDouble("range_weight", 0.2));
    max_detection_speed_ = std::max(0.1, declareOrGetDouble("max_detection_speed", 12.0));
    detection_publish_rate_hz_ = std::max(1.0, declareOrGetDouble("detection_publish_rate_hz", 20.0));
  }

  double declareOrGetDouble(const std::string & name, double default_value)
  {
    if (!has_parameter(name)) {
      return declare_parameter<double>(name, default_value);
    }
    return get_parameter(name).as_double();
  }

  int64_t declareOrGetInt(const std::string & name, int64_t default_value)
  {
    if (!has_parameter(name)) {
      return declare_parameter<int64_t>(name, default_value);
    }
    return get_parameter(name).as_int();
  }

  bool declareOrGetBool(const std::string & name, bool default_value)
  {
    if (!has_parameter(name)) {
      return declare_parameter<bool>(name, default_value);
    }
    return get_parameter(name).as_bool();
  }

  std::string resolveWaypointPath(const std::string & path, bool absolute)
  {
    if (path.empty() || absolute || (!path.empty() && path.front() == '/')) {
      return path;
    }
    const auto share_dir = ament_index_cpp::get_package_share_directory("mppi_bringup");
    return share_dir + "/" + path;
  }

  void loadWaypoints(const std::string & path)
  {
    if (path.empty()) {
      RCLCPP_WARN(get_logger(), "No waypoint_path supplied; detector will not use raceline scoring.");
      return;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
      RCLCPP_ERROR(get_logger(), "Failed to open detector waypoint_path: %s", path.c_str());
      return;
    }

    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      const auto fields = splitSemicolonLine(line);
      if (fields.size() < 4) {
        continue;
      }
      try {
        Waypoint wp;
        wp.s = std::stod(fields[0]);
        wp.x = std::stod(fields[1]);
        wp.y = std::stod(fields[2]);
        wp.yaw = std::stod(fields[3]);
        waypoints_.push_back(wp);
      } catch (const std::exception &) {
        continue;
      }
    }

    if (waypoints_.size() < 2) {
      RCLCPP_ERROR(get_logger(), "Waypoint file has fewer than two usable points: %s", path.c_str());
      waypoints_.clear();
      return;
    }

    const auto & first = waypoints_.front();
    const auto & last = waypoints_.back();
    track_length_ = last.s + std::hypot(first.x - last.x, first.y - last.y);
    if (track_length_ <= 0.0) {
      RCLCPP_ERROR(get_logger(), "Invalid track length from waypoint_path: %s", path.c_str());
      waypoints_.clear();
      track_length_ = 0.0;
    }
  }

  double normalizeS(double s) const
  {
    if (track_length_ <= 0.0) {
      return s;
    }
    s = std::fmod(s, track_length_);
    if (s < 0.0) {
      s += track_length_;
    }
    return s;
  }

  double unwrapS(double measured_s, double reference_s) const
  {
    if (track_length_ <= 0.0) {
      return measured_s;
    }
    double unwrapped = measured_s;
    while (unwrapped - reference_s > 0.5 * track_length_) {
      unwrapped -= track_length_;
    }
    while (unwrapped - reference_s < -0.5 * track_length_) {
      unwrapped += track_length_;
    }
    return unwrapped;
  }

  TrackProjection projectToTrack(double x, double y) const
  {
    TrackProjection best;
    if (waypoints_.size() < 2) {
      best.x = x;
      best.y = y;
      best.yaw = ego_yaw_;
      best.distance = 0.0;
      return best;
    }

    for (std::size_t i = 0; i < waypoints_.size(); ++i) {
      const Waypoint & a = waypoints_[i];
      const Waypoint & b = (i + 1 < waypoints_.size()) ? waypoints_[i + 1] : waypoints_.front();
      const double b_s = (i + 1 < waypoints_.size()) ? b.s : track_length_;
      const double dx = b.x - a.x;
      const double dy = b.y - a.y;
      const double len2 = std::max(kEps, dx * dx + dy * dy);
      const double t = std::clamp(((x - a.x) * dx + (y - a.y) * dy) / len2, 0.0, 1.0);
      const double px = a.x + t * dx;
      const double py = a.y + t * dy;
      const double ex = x - px;
      const double ey = y - py;
      const double dist = std::hypot(ex, ey);

      if (dist < best.distance) {
        best.distance = dist;
        best.s = normalizeS(a.s + t * (b_s - a.s));
        best.x = px;
        best.y = py;
        best.yaw = std::atan2(dy, dx);
        const double nx = -std::sin(best.yaw);
        const double ny = std::cos(best.yaw);
        best.lateral_error = ex * nx + ey * ny;
      }
    }
    return best;
  }

  bool pointNearStaticMap(double x, double y) const
  {
    if (!map_received_) {
      return require_static_map_;
    }

    const auto & info = static_map_.info;
    if (info.resolution <= 0.0 || info.width == 0 || info.height == 0) {
      return require_static_map_;
    }

    const int mx = static_cast<int>((x - info.origin.position.x) / info.resolution);
    const int my = static_cast<int>((y - info.origin.position.y) / info.resolution);
    if (mx < 0 || mx >= static_cast<int>(info.width) || my < 0 || my >= static_cast<int>(info.height)) {
      return true;
    }

    const int inflation_cells = std::max(0, static_cast<int>(std::ceil(wall_inflation_radius_ / info.resolution)));
    for (int dy = -inflation_cells; dy <= inflation_cells; ++dy) {
      const int yy = my + dy;
      if (yy < 0 || yy >= static_cast<int>(info.height)) {
        continue;
      }
      for (int dx = -inflation_cells; dx <= inflation_cells; ++dx) {
        const int xx = mx + dx;
        if (xx < 0 || xx >= static_cast<int>(info.width)) {
          continue;
        }
        const int8_t value = static_map_.data[yy * info.width + xx];
        if (value >= occupied_threshold_ || (treat_unknown_as_static_ && value < 0)) {
          return true;
        }
      }
    }
    return false;
  }

  std::vector<std::vector<int>> clusterPoints(const std::vector<ScanPoint> & points) const
  {
    std::vector<std::vector<int>> clusters;
    std::vector<bool> visited(points.size(), false);
    const double tol2 = cluster_tolerance_ * cluster_tolerance_;

    for (std::size_t i = 0; i < points.size(); ++i) {
      if (visited[i]) {
        continue;
      }

      std::vector<int> cluster;
      std::queue<int> q;
      visited[i] = true;
      q.push(static_cast<int>(i));

      while (!q.empty()) {
        const int idx = q.front();
        q.pop();
        cluster.push_back(idx);

        for (std::size_t j = 0; j < points.size(); ++j) {
          if (visited[j]) {
            continue;
          }
          const double dx = points[idx].map_x - points[j].map_x;
          const double dy = points[idx].map_y - points[j].map_y;
          if (dx * dx + dy * dy <= tol2) {
            visited[j] = true;
            q.push(static_cast<int>(j));
          }
        }
      }
      clusters.push_back(cluster);
    }
    return clusters;
  }

  bool buildCandidate(
    const std::vector<int> & cluster,
    const std::vector<ScanPoint> & points,
    const rclcpp::Time & stamp,
    ClusterCandidate & candidate) const
  {
    if (
      cluster.size() < static_cast<std::size_t>(min_cluster_points_) ||
      cluster.size() > static_cast<std::size_t>(max_cluster_points_))
    {
      return false;
    }

    for (const int idx : cluster) {
      candidate.centroid_x += points[idx].map_x;
      candidate.centroid_y += points[idx].map_y;
      candidate.centroid_base_x += points[idx].base_x;
      candidate.centroid_base_y += points[idx].base_y;
    }
    const double inv_n = 1.0 / static_cast<double>(cluster.size());
    candidate.centroid_x *= inv_n;
    candidate.centroid_y *= inv_n;
    candidate.centroid_base_x *= inv_n;
    candidate.centroid_base_y *= inv_n;
    candidate.indices = cluster;
    candidate.range = std::hypot(candidate.centroid_base_x, candidate.centroid_base_y);

    if (candidate.centroid_base_x < min_base_x_) {
      return false;
    }

    const TrackProjection proj = projectToTrack(candidate.centroid_x, candidate.centroid_y);
    if (proj.distance > max_raceline_projection_dist_) {
      return false;
    }

    const double tx = std::cos(proj.yaw);
    const double ty = std::sin(proj.yaw);
    const double nx = -std::sin(proj.yaw);
    const double ny = std::cos(proj.yaw);

    double min_t = std::numeric_limits<double>::infinity();
    double max_t = -std::numeric_limits<double>::infinity();
    double min_n = std::numeric_limits<double>::infinity();
    double max_n = -std::numeric_limits<double>::infinity();
    for (const int idx : cluster) {
      const double dx = points[idx].map_x - candidate.centroid_x;
      const double dy = points[idx].map_y - candidate.centroid_y;
      const double local_t = dx * tx + dy * ty;
      const double local_n = dx * nx + dy * ny;
      min_t = std::min(min_t, local_t);
      max_t = std::max(max_t, local_t);
      min_n = std::min(min_n, local_n);
      max_n = std::max(max_n, local_n);
    }
    candidate.extent_t = std::max(0.0, max_t - min_t);
    candidate.extent_n = std::max(0.0, max_n - min_n);
    const double max_extent = std::max(candidate.extent_t, candidate.extent_n);
    if (max_extent < min_cluster_extent_ || max_extent > max_cluster_extent_) {
      return false;
    }

    const double view_x = candidate.centroid_x - ego_x_;
    const double view_y = candidate.centroid_y - ego_y_;
    const double view_norm = std::max(kEps, std::hypot(view_x, view_y));
    const double ux = view_x / view_norm;
    const double uy = view_y / view_norm;
    const double dir_t = ux * tx + uy * ty;
    const double dir_n = ux * nx + uy * ny;
    const double missing_t = std::max(0.0, 0.5 * opponent_length_ - 0.5 * candidate.extent_t);
    const double missing_n = std::max(0.0, 0.5 * opponent_width_ - 0.5 * candidate.extent_n);
    double corr_x = tx * dir_t * missing_t + nx * dir_n * missing_n;
    double corr_y = ty * dir_t * missing_t + ny * dir_n * missing_n;
    candidate.correction_norm = std::hypot(corr_x, corr_y);
    if (candidate.correction_norm > max_center_correction_ && candidate.correction_norm > kEps) {
      const double scale = max_center_correction_ / candidate.correction_norm;
      corr_x *= scale;
      corr_y *= scale;
      candidate.correction_norm = max_center_correction_;
    }

    candidate.center_x = candidate.centroid_x + corr_x;
    candidate.center_y = candidate.centroid_y + corr_y;
    const TrackProjection corrected_proj = projectToTrack(candidate.center_x, candidate.center_y);
    candidate.center_s = corrected_proj.s;
    candidate.center_yaw = corrected_proj.yaw;
    candidate.projection_distance = corrected_proj.distance;

    if (candidate.projection_distance > max_raceline_projection_dist_) {
      return false;
    }

    candidate.score =
      projection_weight_ * candidate.projection_distance +
      range_weight_ * candidate.range;

    if (has_last_detection_) {
      const double age = (stamp - last_detection_stamp_).seconds();
      const double jump = std::hypot(
        candidate.center_x - last_detection_x_,
        candidate.center_y - last_detection_y_);
      if (age >= 1e-3 && age < continuity_gate_timeout_) {
        const double candidate_s = unwrapS(candidate.center_s, last_detection_s_);
        const double implied_progress_speed = std::abs(candidate_s - last_detection_s_) / age;
        if (implied_progress_speed > max_detection_speed_) {
          return false;
        }
      }
      if (age >= 0.0 && age < continuity_gate_timeout_ && jump > max_candidate_jump_) {
        return false;
      }
      if (age >= 0.0 && age < continuity_gate_timeout_) {
        candidate.score += continuity_weight_ * jump;
      }
    }

    return true;
  }

  bool selectCandidate(
    const std::vector<ScanPoint> & points,
    const std::vector<std::vector<int>> & clusters,
    const rclcpp::Time & stamp,
    ClusterCandidate & selected) const
  {
    bool found = false;
    for (const auto & cluster : clusters) {
      ClusterCandidate candidate;
      if (!buildCandidate(cluster, points, stamp, candidate)) {
        continue;
      }
      if (!found || candidate.score < selected.score) {
        selected = candidate;
        found = true;
      }
    }
    return found;
  }

  bool shouldPublishDetection(const rclcpp::Time & stamp)
  {
    if (!has_last_detection_publish_) {
      return true;
    }
    const double min_dt = 1.0 / detection_publish_rate_hz_;
    return (stamp - last_detection_publish_stamp_).seconds() >= min_dt;
  }

  void publishDetection(const ClusterCandidate & candidate, const rclcpp::Time & stamp)
  {
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = frame_id_;
    odom.child_frame_id = "opponent_lidar_detection";
    odom.pose.pose.position.x = candidate.center_x;
    odom.pose.pose.position.y = candidate.center_y;
    odom.pose.pose.orientation = yawToQuaternion(candidate.center_yaw);

    double speed = 0.0;
    if (has_last_detection_) {
      const double dt = (stamp - last_detection_stamp_).seconds();
      if (dt > 1e-3) {
        const double unwrapped_s = unwrapS(candidate.center_s, last_detection_s_);
        speed = (unwrapped_s - last_detection_s_) / dt;
        speed = std::clamp(speed, 0.0, max_detection_speed_);
      }
    }
    odom.twist.twist.linear.x = speed;
    detection_pub_->publish(odom);

    last_detection_x_ = candidate.center_x;
    last_detection_y_ = candidate.center_y;
    last_detection_s_ = has_last_detection_ ? unwrapS(candidate.center_s, last_detection_s_) : candidate.center_s;
    last_detection_stamp_ = stamp;
    has_last_detection_ = true;
    last_detection_publish_stamp_ = stamp;
    has_last_detection_publish_ = true;
  }

  void publishMarkers(
    const std::vector<ScanPoint> & dynamic_points,
    const ClusterCandidate * selected,
    const rclcpp::Time & stamp) const
  {
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear);

    visualization_msgs::msg::Marker dyn;
    dyn.header.stamp = stamp;
    dyn.header.frame_id = frame_id_;
    dyn.ns = "opponent_lidar_detection";
    dyn.id = 0;
    dyn.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    dyn.action = visualization_msgs::msg::Marker::ADD;
    dyn.scale.x = 0.06;
    dyn.scale.y = 0.06;
    dyn.scale.z = 0.06;
    dyn.color.a = 0.75;
    dyn.color.r = 1.0;
    dyn.color.g = 0.1;
    dyn.color.b = 0.1;
    for (const auto & p : dynamic_points) {
      geometry_msgs::msg::Point point;
      point.x = p.map_x;
      point.y = p.map_y;
      point.z = 0.08;
      dyn.points.push_back(point);
    }
    markers.markers.push_back(dyn);

    if (selected != nullptr) {
      visualization_msgs::msg::Marker center;
      center.header.stamp = stamp;
      center.header.frame_id = frame_id_;
      center.ns = "opponent_lidar_detection";
      center.id = 1;
      center.type = visualization_msgs::msg::Marker::ARROW;
      center.action = visualization_msgs::msg::Marker::ADD;
      center.pose.position.x = selected->center_x;
      center.pose.position.y = selected->center_y;
      center.pose.position.z = 0.2;
      center.pose.orientation = yawToQuaternion(selected->center_yaw);
      center.scale.x = opponent_length_;
      center.scale.y = opponent_width_;
      center.scale.z = 0.12;
      center.color.a = 0.9;
      center.color.r = 0.1;
      center.color.g = 1.0;
      center.color.b = 0.25;
      markers.markers.push_back(center);

      visualization_msgs::msg::Marker text;
      text.header.stamp = stamp;
      text.header.frame_id = frame_id_;
      text.ns = "opponent_lidar_detection";
      text.id = 2;
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose.position.x = selected->center_x;
      text.pose.position.y = selected->center_y;
      text.pose.position.z = 0.65;
      text.scale.z = 0.2;
      text.color.a = 0.95;
      text.color.r = 1.0;
      text.color.g = 1.0;
      text.color.b = 1.0;
      std::ostringstream ss;
      ss << "det score=" << std::fixed << std::setprecision(2) << selected->score
         << " d=" << selected->projection_distance;
      text.text = ss.str();
      markers.markers.push_back(text);
    }

    marker_pub_->publish(markers);
  }

  void publishDebug(
    std::size_t dynamic_count,
    std::size_t cluster_count,
    const ClusterCandidate * selected,
    double progress_speed) const
  {
    std_msgs::msg::Float32MultiArray debug;
    debug.data = {
      static_cast<float>(dynamic_count),
      static_cast<float>(cluster_count),
      selected ? static_cast<float>(selected->score) : -1.0f,
      selected ? static_cast<float>(selected->range) : -1.0f,
      selected ? static_cast<float>(selected->projection_distance) : -1.0f,
      selected ? static_cast<float>(selected->correction_norm) : -1.0f,
      selected ? static_cast<float>(selected->extent_t) : -1.0f,
      selected ? static_cast<float>(selected->extent_n) : -1.0f,
      static_cast<float>(progress_speed),
      map_received_ ? 1.0f : 0.0f
    };
    debug_pub_->publish(debug);
  }

  void egoOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    ego_x_ = msg->pose.pose.position.x;
    ego_y_ = msg->pose.pose.position.y;
    ego_yaw_ = yawFromQuaternion(msg->pose.pose.orientation);
    ego_stamp_ = msg->header.stamp;
    have_ego_ = true;
  }

  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    static_map_ = *msg;
    map_received_ = true;
  }

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    refreshLiveParams();

    if (!have_ego_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Waiting for ego odom.");
      return;
    }
    if (require_static_map_ && !map_received_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Waiting for static map.");
      return;
    }

    const rclcpp::Time stamp = msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0 ?
      now() : rclcpp::Time(msg->header.stamp);

    const double cy = std::cos(ego_yaw_);
    const double sy = std::sin(ego_yaw_);
    const double cl = std::cos(laser_yaw_offset_);
    const double sl = std::sin(laser_yaw_offset_);
    std::vector<ScanPoint> dynamic_points;
    dynamic_points.reserve(msg->ranges.size());

    for (std::size_t i = 0; i < msg->ranges.size(); ++i) {
      const double r = msg->ranges[i];
      if (!std::isfinite(r) || r < min_range_ || r > max_range_ || r > msg->range_max) {
        continue;
      }
      const double angle = msg->angle_min + static_cast<double>(i) * msg->angle_increment;
      if (front_fov_only_ && std::abs(angle) > 0.5 * M_PI) {
        continue;
      }

      const double lx = r * std::cos(angle);
      const double ly = r * std::sin(angle);
      const double bx = laser_x_offset_ + cl * lx - sl * ly;
      const double by = laser_y_offset_ + sl * lx + cl * ly;
      const double base_range = std::hypot(bx, by);
      if (base_range < min_detection_range_ || base_range > max_detection_range_ || bx < min_base_x_) {
        continue;
      }

      ScanPoint point;
      point.base_x = bx;
      point.base_y = by;
      point.map_x = ego_x_ + cy * bx - sy * by;
      point.map_y = ego_y_ + sy * bx + cy * by;
      if (pointNearStaticMap(point.map_x, point.map_y)) {
        continue;
      }
      dynamic_points.push_back(point);
    }

    const auto clusters = clusterPoints(dynamic_points);
    ClusterCandidate selected;
    const bool found = selectCandidate(dynamic_points, clusters, stamp, selected);
    double progress_speed = 0.0;
    if (found && has_last_detection_) {
      const double dt = (stamp - last_detection_stamp_).seconds();
      if (dt > 1e-3) {
        progress_speed = std::clamp(
          (unwrapS(selected.center_s, last_detection_s_) - last_detection_s_) / dt,
          0.0,
          max_detection_speed_);
      }
    }

    if (found) {
      if (shouldPublishDetection(stamp)) {
        publishDetection(selected, stamp);
      }
      publishMarkers(dynamic_points, &selected, stamp);
      publishDebug(dynamic_points.size(), clusters.size(), &selected, progress_speed);
    } else {
      publishMarkers(dynamic_points, nullptr, stamp);
      publishDebug(dynamic_points.size(), clusters.size(), nullptr, 0.0);
    }
  }

  std::string scan_topic_;
  std::string ego_odom_topic_;
  std::string map_topic_;
  std::string detected_odom_topic_;
  std::string detector_marker_topic_;
  std::string detector_debug_topic_;
  std::string waypoint_path_;
  bool waypoint_path_absolute_ = false;
  std::string frame_id_ = "map";
  std::string base_frame_id_ = "ego_racecar/base_link";

  double laser_x_offset_ = 0.0;
  double laser_y_offset_ = 0.0;
  double laser_yaw_offset_ = 0.0;
  double opponent_length_ = 0.58;
  double opponent_width_ = 0.31;
  double max_center_correction_ = 0.35;
  double min_range_ = 0.15;
  double max_range_ = 8.0;
  double min_detection_range_ = 0.35;
  double max_detection_range_ = 6.0;
  bool front_fov_only_ = false;
  double min_base_x_ = -0.5;
  bool require_static_map_ = true;
  int occupied_threshold_ = 50;
  double wall_inflation_radius_ = 0.18;
  bool treat_unknown_as_static_ = true;
  double cluster_tolerance_ = 0.18;
  int min_cluster_points_ = 4;
  int max_cluster_points_ = 80;
  double min_cluster_extent_ = 0.08;
  double max_cluster_extent_ = 0.9;
  double max_raceline_projection_dist_ = 1.2;
  double max_candidate_jump_ = 1.5;
  double continuity_gate_timeout_ = 0.4;
  double continuity_weight_ = 2.0;
  double projection_weight_ = 1.0;
  double range_weight_ = 0.2;
  double max_detection_speed_ = 12.0;
  double detection_publish_rate_hz_ = 20.0;

  bool have_ego_ = false;
  double ego_x_ = 0.0;
  double ego_y_ = 0.0;
  double ego_yaw_ = 0.0;
  rclcpp::Time ego_stamp_;

  bool map_received_ = false;
  nav_msgs::msg::OccupancyGrid static_map_;

  std::vector<Waypoint> waypoints_;
  double track_length_ = 0.0;

  bool has_last_detection_ = false;
  double last_detection_x_ = 0.0;
  double last_detection_y_ = 0.0;
  double last_detection_s_ = 0.0;
  rclcpp::Time last_detection_stamp_;
  bool has_last_detection_publish_ = false;
  rclcpp::Time last_detection_publish_stamp_;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ego_odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr detection_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr debug_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OpponentLidarDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
