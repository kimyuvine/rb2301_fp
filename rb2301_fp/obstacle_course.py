import argparse
import configparser
import heapq
import math
import os
import time
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import ReliabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
set_logger_level("obstaclecourse", level=LoggingSeverity.INFO) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG  


_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

def wrap_deg(angle:float) -> float:
    '''Wrap an angle in degrees into [-180, 180).'''
    return (angle + 180.0) % 360.0 - 180.0

def optitrack_to_maze(x:float, y:float, heading_deg:float, frame:dict) -> tuple:
    '''Optitrack-frame pose (x, y, heading in degrees) -> CA2 maze-frame pose (see optitrack_variables.config).'''
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame["heading_offset_deg"]))

def maze_to_course(x:float, y:float, heading_deg:float, course:dict) -> tuple:
    '''CA2 maze-frame pose -> final-project course frame (origin/rotation of the course in the maze frame, [course] in the config).'''
    dx, dy = x - course["origin_x"], y - course["origin_y"]
    r = math.radians(course["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy, wrap_deg(heading_deg - course["rotation_deg"]))

def load_frame(robot_number:int) -> tuple:
    '''Read optitrack_variables.config: returns (frame, course) for this robot.'''
    parser = configparser.ConfigParser()
    path = os.path.join(_PACKAGE_DIR, "optitrack_variables.config")
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read {path}")
    frame = {k: parser.getfloat("frame", k) for k in ("origin_x", "origin_y", "rotation_deg")}
    raw = parser.get("frame", "heading_offset_deg", fallback="auto").strip().lower()
    # Rigid bodies are created facing +y of the maze frame, so Motive's zero heading is maze +y (90 deg)
    frame["heading_offset_deg"] = wrap_deg(-(frame["rotation_deg"] + 90.0)) if raw == "auto" else float(raw)
    if parser.has_section("heading_offsets") and parser.has_option("heading_offsets", str(robot_number)):
        frame["heading_offset_deg"] = parser.getfloat("heading_offsets", str(robot_number))
    course = {k: parser.getfloat("course", k, fallback=0.0) for k in ("origin_x", "origin_y", "rotation_deg")}
    return frame, course


# --- Final project: parameters ----------------------------------------------
# Robot footprint, measured from the pose point, which is the REAR axle (see nanocar_description.urdf)
ROBOT_REAR = 0.05           # pose point -> back bumper (m)
ROBOT_FRONT = 0.20          # pose point -> front bumper (m)
ROBOT_HALF_WIDTH = 0.11     # m
LIDAR_X = 0.114             # pose point -> LiDAR (m)
CENTRE_X = (ROBOT_FRONT - ROBOT_REAR) / 2       # pose point -> centre of the footprint (m)
HALF_LENGTH = (ROBOT_FRONT + ROBOT_REAR) / 2    # m
PLATE_HALF_T = 0.025        # half the thickness of a swinging plate (m)

# Occupancy grid used by the planner
GRID_RES = 0.05             # m per cell; the gaps beside the static obstacles can be under 0.4 m
BLOCK_RADIUS = 0.13         # cells whose centre is closer than this to a LiDAR hit are off-limits for the footprint centre:
                            # half the robot's width + 2 cm, so a 0.3 m gap is just passable and a 0.25 m one is not
SOFT_RADIUS = 0.35          # between BLOCK_RADIUS and this, cells cost extra so paths keep to the middle
PROX_COST = 3.0             # extra step cost right at BLOCK_RADIUS
SELF_MARGIN = 0.02          # LiDAR returns inside the footprint grown by this are the robot itself (marker posts, cables): dropped

LIDAR_CHECK = False         # True: never drive, just print where the closest LiDAR return is (check ranges[0] is straight ahead on a new robot)
DEBUG = True                # True: extra "DBG" prints -- why it replans, how it is driving, what the safety layer saw
DEBUG_MAP_PERIOD = 5.0      # s between ASCII maps of the area around the robot (also printed on entering RECOVER); 0 = never

# Everything that differs between Gazebo and the lab. Positions are in the COURSE frame: origin at the start point,
# +x up the first corridor, +y to its left (in Gazebo this is the world frame). None of these need to be exact.
SIM_CONFIG = {
    "bounds": (-0.8, 5.8, -4.6, 1.0),       # xmin, xmax, ymin, ymax of the area the course can occupy (the map is only built inside it)
    "goal": (4.4, -2.6),                    # roughly past the swinging obstacles; the planner goes as close as it can
    "door_region": (2.7, 3.7, -3.4, -2.1),  # where the centre of the door can be (xmin, xmax, ymin, ymax); None = anywhere. The randomiser
                                            # puts the door line at x 2.8-3.6, gap centre y -3.3..-2.2; obstacle faces never pass x 2.6
    "door_width": (0.34, 0.55),             # m, gaps accepted as the door (the gate gap is 0.4 m and measures 0.40-0.45)
    "door_support": (0.15, 0.55),           # m of unbroken wall needed on the shorter / longer side of the gap (a drawer face is 0.48 m)
    "door_stage_gap": 0.08,                 # m between the front bumper and the door wall where the robot lines up with the gap
    "plate_spacing": 0.6,                  # door -> plate line 1 -> plate line 2, only used until the plates are seen
    "lidar_start_deg": np.rad2deg(-3.14 + 3.1416),  # direction of ranges[0] in the robot frame (0 = ahead, CCW +ve): scan starts at -3.14 rad, LiDAR mounted turned by pi
    "lidar_step_deg": np.rad2deg(6.28 / 359),       # 360 samples from -3.14 to 3.14 rad; None = 360 / number of samples
    "lidar_ccw": True,                      # False if the scan goes clockwise (LIDAR_CHECK: an object on the robot's left reads -90 instead of +90)
    "cruise_speed": 0.5,                    # m/s along the A* path
    "door_speed": 0.2,                      # m/s when lining up with / going through the door
    "dash_speed": 1.0,                      # m/s past a swinging plate
    "strafe_speed": 0.25,                   # m/s sideways (robot frame); every move is a slide now, so this limits all sideways motion
    "backup_speed": 0.15,                   # m/s when recovering
    "min_run": 0.0,                         # smallest forward/sideways speed that actually moves the robot
    "turn_kp": 2.0, "turn_kd": 0.05,        # heading PID (rad/s per rad of error)
    "max_turn": 2.0, "min_turn": 0.25,      # rad/s
    "heading_hold_deg": 10.0,               # never turns to steer; only turns back if the heading drifts more than this off course +x (None = never turn)
    "lookahead": 0.35,                      # m, path-following look-ahead
    "replan_period": 0.5,                   # s between A* replans (it also replans as soon as the path gets blocked)
    "refresh_period": 0.25,                 # s between recomputing the map's distance field (~10 ms each on a laptop)
    "cross_latency": 0.2,                   # s added to the time needed to clear a plate line
    "cross_safety": 1.3,                    # the plate must need this many times our crossing time to reach our lane
    "plate_speed_floor": 0.3,               # m/s assumed for a plate that has not been seen moving yet
    "unseen_timeout": 3.0,                  # s to wait for a plate that is never seen before going anyway (if the way is clear)
    "settle_ticks": 20,                     # ticks to wait at the start (like CA2's delay_counter)
}
REAL_CONFIG = dict(SIM_CONFIG, **{
    "bounds": (-1.0, 6.5, -5.0, 1.2),       # TODO measure the lab course (./fp.sh --calibrate) and tighten these
    "goal": (4.4, -2.6),                    # TODO set from the lab course
    "door_region": None,                    # TODO set from the lab course to reject door look-alikes (e.g. two boxes side by side)
    "door_width": (0.34, 0.8),             # TODO tighten around the lab door's width (logged when DOOR_ALIGN starts)
    "door_stage_gap": 0.10,                 # a little more room for Optitrack noise and min_run overshoot
    "lidar_start_deg": 0.0,                 # check with LIDAR_CHECK = True
    "lidar_step_deg": None,
    "cruise_speed": 0.6, "door_speed": 0.1, "dash_speed": 1.0, "strafe_speed": 0.3, "backup_speed": 0.08,
    "min_run": 0.03,
    "turn_kp": 1.5, "turn_kd": 0.05, "max_turn": 0.6, "min_turn": 0.2,
    "lookahead": 0.3,
    "replan_period": 1.0,
    "refresh_period": 0.5,
    "cross_latency": 0.5,
    "cross_safety": 1.6,
    "plate_speed_floor": 0.8,               # a real plate may be let go at any moment
    "unseen_timeout": 8.0,
})


# --- Final project: perception helpers ----------------------------------------
def lidar_points(ranges:np.ndarray, start_deg:float, step_deg:float) -> tuple:
    '''LiDAR ranges -> (end points in the robot frame (N, 2), valid mask (N,), unit ray directions (N, 2), ranges (N,)).
    Robot frame: origin at the pose point, +x forward, +y left. Invalid returns (0, inf, nan, too close) are masked out.'''
    r = np.asarray(ranges, dtype=float)
    a = np.deg2rad(start_deg + step_deg * np.arange(len(r)))
    dirs = np.column_stack((np.cos(a), np.sin(a)))
    valid = np.isfinite(r) & (r > 0.06) & (r < 8.0)
    points = np.array((LIDAR_X, 0.0)) + dirs * np.where(valid, r, 0.0)[:, None]
    return points, valid, dirs, r

def to_frame(points_r:np.ndarray, pose) -> np.ndarray:
    '''Robot-frame points (N, 2) -> the frame the pose (x, y, heading in degrees) is given in.'''
    h = math.radians(pose[2])
    c, s = math.cos(h), math.sin(h)
    return np.column_stack((pose[0] + c * points_r[:, 0] - s * points_r[:, 1], pose[1] + s * points_r[:, 0] + c * points_r[:, 1]))

def estimate_wall_axis(points_r:np.ndarray) -> tuple:
    '''Walls are rectilinear, so segments between nearby scan points mostly point along one of two perpendicular axes.
    The segments join points a few apart (0.15-0.5 m) rather than neighbours: neighbours are only 1-2 cm apart, and 1 cm
    of range noise would make their directions random. Returns (direction of the axis nearest to straight ahead in
    degrees, robot frame, in (-45, 45], fraction of segments that agree with it).'''
    d = np.vstack([points_r[s:] - points_r[:-s] for s in (4, 8, 16) if len(points_r) > s])
    length = np.hypot(d[:, 0], d[:, 1])
    d = d[(length > 0.15) & (length < 0.5)]
    if len(d) < 20:
        return 0.0, 0.0
    ang = np.rad2deg(np.arctan2(d[:, 1], d[:, 0])) % 90.0
    hist = np.bincount(ang.astype(int) % 90, minlength=90).astype(float)
    k = int(np.argmax(hist + np.roll(hist, 1) + np.roll(hist, -1)))
    diff = (ang - (k + 0.5) + 45.0) % 90.0 - 45.0
    near = np.abs(diff) < 2.5
    peak = (k + 0.5 + diff[near].mean()) % 90.0
    return (peak if peak <= 45.0 else peak - 90.0), float(near.mean())

def free_distance(points_r:np.ndarray, ux:float, uy:float, margin:float=0.02) -> float:
    '''How far (m) the footprint (grown by margin) can slide along the unit direction (ux, uy) (robot frame) before it
    touches a LiDAR point; inf if never. Points already inside the footprint (e.g. the LiDAR seeing the robot itself)
    are ignored. Per point, the slide distances that put it inside the footprint form an interval on each axis.'''
    if len(points_r) == 0:
        return math.inf
    lo, hi = np.full(len(points_r), -np.inf), np.full(len(points_r), np.inf)
    for axis, u, a, b in ((0, ux, -ROBOT_REAR - margin, ROBOT_FRONT + margin), (1, uy, -ROBOT_HALF_WIDTH - margin, ROBOT_HALF_WIDTH + margin)):
        p = points_r[:, axis]
        if abs(u) < 1e-9:
            outside = (p < a) | (p > b)
            lo[outside], hi[outside] = np.inf, -np.inf
        else:
            s1, s2 = (p - b) / u, (p - a) / u
            lo, hi = np.maximum(lo, np.minimum(s1, s2)), np.minimum(hi, np.maximum(s1, s2))
    ahead = (lo <= hi) & (lo >= 0.0)
    return float(lo[ahead].min()) if np.any(ahead) else math.inf

def swept_clear(points_r:np.ndarray, dx:float, dy:float, margin:float=0.02) -> bool:
    '''True if sliding the footprint by (dx, dy) (robot frame, metres) would not touch any LiDAR point.'''
    d = math.hypot(dx, dy)
    return d < 1e-9 or free_distance(points_r, dx / d, dy / d, margin) > d

def rotation_clear(points_r:np.ndarray, angle_deg:float, margin:float=0.01) -> bool:
    '''True if turning on the spot (about the pose point, as the robot does) by angle_deg would not sweep the footprint
    into any LiDAR point. Points already inside the footprint are ignored.'''
    x0, x1, hw = -ROBOT_REAR - margin, ROBOT_FRONT + margin, ROBOT_HALF_WIDTH + margin
    inside = lambda q: (q[:, 0] > x0) & (q[:, 0] < x1) & (np.abs(q[:, 1]) < hw)
    p = points_r[~inside(points_r)]
    for f in (1 / 3, 2 / 3, 1.0): # turning the robot by +a = turning the points by -a
        a = -math.radians(angle_deg * f)
        c, s = math.cos(a), math.sin(a)
        if np.any(inside(np.column_stack((c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1])))):
            return False
    return True

def side_clearance(points_r:np.ndarray) -> tuple:
    '''(left, right) gap in metres between the sides of the footprint and the nearest LiDAR point beside it.'''
    beside = (points_r[:, 0] > -ROBOT_REAR) & (points_r[:, 0] < ROBOT_FRONT)
    y = points_r[beside, 1]
    left, right = y[y > 0], -y[y < 0]
    return (left.min() - ROBOT_HALF_WIDTH if len(left) else math.inf,
            right.min() - ROBOT_HALF_WIDTH if len(right) else math.inf)

def wall_run(ys:np.ndarray, start:float, direction:int, max_step:float=0.1) -> float:
    '''Length of unbroken wall (sorted point coordinates ys, no jumps over max_step) from `start` going `direction` (+1/-1).'''
    side = ys[ys >= start] if direction > 0 else ys[ys <= start][::-1]
    if len(side) == 0:
        return 0.0
    steps = np.abs(np.diff(side))
    end = int(np.argmax(steps > max_step)) if np.any(steps > max_step) else len(side) - 1
    return float(abs(side[end] - start))

def find_door(points:np.ndarray, sensor:np.ndarray, cx:float, cy:float, region=None, ahead=(0.1, 1.6),
              half_window:float=1.2, width=(0.4, 0.8), support=(0.15, 0.55), max_tilt:float=12.0, why:list=None):
    '''Look for the door: a wall across the course +x direction with a gap in it that the LiDAR can see through. The wall
    may be up to max_tilt deg off square (course frame set a little off the real walls, or walls not quite square): each
    tilt is tried, square first, by turning the points about the robot so that the wall becomes a line of constant x.
    One side of the gap needs support[1] m of unbroken wall (two boxes side by side make a gap too, but they are shorter
    than that); the other support[0] m, or a wall running along x from the gap's edge (a door next to a corridor wall).
    points = course-frame LiDAR points, sensor = LiDAR position, (cx, cy) = robot centre; why = list that collects why
    each gap was turned down (debugging).
    Returns {"x", "y", "width", "runs", "tilt"} of the nearest such gap (course frame), or None.'''
    for tilt in sorted(np.arange(-max_tilt, max_tilt + 1e-6, 3.0), key=abs):
        t = math.radians(tilt)
        m = np.array(((math.cos(t), -math.sin(t)), (math.sin(t), math.cos(t)))) # turns points by tilt about the robot
        to_course = lambda x, y: (np.array((x, y)) - (cx, cy)) @ m + (cx, cy) # and back
        door = door_on_square_wall((points - (cx, cy)) @ m.T + (cx, cy), (sensor - (cx, cy)) @ m.T + (cx, cy), cx, cy,
                                   ahead, half_window, width, support, region, to_course, float(tilt), why)
        if door is not None:
            return door
    return None

def door_on_square_wall(points, sensor, cx, cy, ahead, half_window, width, support, region, to_course, tilt, why):
    '''find_door for one tilt: points and sensor already turned so that the wall to look for is a line of constant x.'''
    rel = points - (cx, cy)
    p = points[(rel[:, 0] > ahead[0]) & (rel[:, 0] < ahead[1]) & (np.abs(rel[:, 1]) < half_window)]
    if len(p) < 10:
        return None
    edges = np.arange(cx + ahead[0], cx + ahead[1] + 0.03, 0.03)
    counts, _ = np.histogram(p[:, 0], edges)
    angles = np.arctan2(points[:, 1] - sensor[1], points[:, 0] - sensor[0])
    for k in range(len(counts)): # nearest wall line first
        if counts[k] < 6 or (k > 0 and counts[k] < counts[k - 1]) or (k + 1 < len(counts) and counts[k] < counts[k + 1]):
            continue
        line_x = float(np.median(p[np.abs(p[:, 0] - (edges[k] + 0.015)) < 0.045, 0]))
        ys = np.sort(p[np.abs(p[:, 0] - line_x) < 0.035, 1])
        best, best_c = None, None
        for g in np.nonzero((np.diff(ys) > width[0]) & (np.diff(ys) < width[1]))[0]:
            lo, hi = ys[g], ys[g + 1]
            centre = (lo + hi) / 2
            x, y = to_course(line_x, centre)
            note = f"tilt {tilt:+.0f}: gap x={x:.2f} y={y:.2f} {hi - lo:.2f} m wide"
            runs = (wall_run(ys, lo, -1), wall_run(ys, hi, +1))
            jambs = [run >= support[0] or side_wall_at(p, line_x, edge) for run, edge in zip(runs, (lo, hi))]
            if not all(jambs) or max(runs) < support[1]:
                if why is not None:
                    why.append(f"{note}, wall {runs[0]:.2f} / {runs[1]:.2f} m either side" + ("" if all(jambs) else ", no corridor wall at the short side"))
                continue
            if region is not None and not (region[0] <= x <= region[1] and region[2] <= y <= region[3]):
                if why is not None:
                    why.append(f"{note}, outside door_region")
                continue
            # Must see through the middle of the gap; a gap that is an obstacle's shadow on the wall has nothing behind it
            a_lo = math.atan2(lo - sensor[1], line_x - sensor[0])
            a_hi = math.atan2(hi - sensor[1], line_x - sensor[0])
            a_min, a_max = min(a_lo, a_hi), max(a_lo, a_hi)
            pad = 0.25 * (a_max - a_min)
            ray_x = points[(angles > a_min + pad) & (angles < a_max - pad), 0]
            through, blocked = np.count_nonzero(ray_x > line_x + 0.1), np.count_nonzero(ray_x < line_x - 0.05)
            if through < 2 or through <= 2 * blocked:
                if why is not None:
                    why.append(f"{note}, can't see through it ({through} rays through, {blocked} stopped short)")
                continue
            if best is None or abs(centre - cy) < abs(best_c - cy):
                best, best_c = {"x": float(x), "y": float(y), "width": float(hi - lo), "runs": runs, "tilt": tilt}, centre
        if best is not None:
            return best
    return None

def side_wall_at(points:np.ndarray, line_x:float, y:float, reach:float=0.5) -> bool:
    '''True if a wall runs along x from the door line at y, beyond the door (a corridor wall that the door opens onto):
    points sharing that y spread over at least 0.2 m of x within `reach` past the line. A wall on the robot's side of the
    line doesn't count: that is just a corner of the corridor the robot is in.'''
    q = points[(np.abs(points[:, 1] - y) < 0.04) & (points[:, 0] > line_x + 0.05) & (points[:, 0] < line_x + reach)]
    return len(np.unique(np.floor(q[:, 0] / 0.05))) >= 4

def side_walls(points:np.ndarray, cx:float, cy:float, x_range=(-0.4, 1.5), max_side:float=1.5) -> list:
    '''[y of the nearest wall on the left, y of the nearest wall on the right] (walls running along x), or None for
    a side with no wall seen. A wall shows up as many points sharing one y, spread along x (points in at least five
    different 0.1 m stretches): a plate seen up close also gives many points per y, but all at one x.'''
    rel = points - (cx, cy)
    sel = (rel[:, 0] > x_range[0]) & (rel[:, 0] < x_range[1])
    out = []
    for side in (1, -1):
        p = points[sel & (side * rel[:, 1] > 0.12) & (side * rel[:, 1] < max_side)]
        found = None
        if len(p) >= 6:
            edges = np.arange(p[:, 1].min(), p[:, 1].max() + 0.06, 0.03)
            counts, _ = np.histogram(p[:, 1], edges)
            for k in sorted(np.nonzero(counts >= 6)[0], key=lambda k: abs(edges[k] - cy)): # nearest first
                band = np.abs(p[:, 1] - (edges[k] + 0.015)) < 0.04
                if len(np.unique(np.floor(p[band, 0] / 0.1))) >= 5:
                    found = float(np.median(p[band, 1]))
                    break
        out.append(found)
    return out

def plate_measurements(points:np.ndarray, x_min:float, x_max:float, y_lo:float, y_hi:float) -> tuple:
    '''Swinging plates are short segments across the corridor (points sharing one x, 0.08-0.6 m long) between the
    side walls. Returns ([(x, y_centre, length) per plate seen], [x of every longer wall across the corridor]).'''
    p = points[(points[:, 0] > x_min) & (points[:, 0] < x_max) & (points[:, 1] > y_lo) & (points[:, 1] < y_hi)]
    plates, walls = [], []
    if len(p) < 3:
        return plates, walls
    counts, edges = np.histogram(p[:, 0], np.arange(x_min, x_max + 0.04, 0.04))
    used = np.zeros(len(p), dtype=bool)
    for k in np.argsort(-counts):
        if counts[k] < 3:
            break
        on = (np.abs(p[:, 0] - (edges[k] + 0.02)) < 0.05) & ~used
        if np.count_nonzero(on) < 3:
            continue
        used |= on
        x_line = float(np.median(p[on, 0]))
        ys = np.sort(p[on, 1])
        if ys[-1] - ys[0] > 0.6:
            walls.append(x_line) # e.g. the end wall, partly hidden behind the plates
            continue
        seg = max(np.split(ys, np.nonzero(np.diff(ys) > 0.08)[0] + 1), key=lambda s: s[-1] - s[0])
        if len(seg) >= 3 and seg[-1] - seg[0] >= 0.08:
            plates.append((x_line, float(seg[0] + seg[-1]) / 2, float(seg[-1] - seg[0])))
    return plates, walls


# --- Final project: mapping and planning (A* from CA2) ----------------------------
SQRT2 = math.sqrt(2.0)
NEIGHBOURS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))

class OccupancyGrid:
    '''Occupancy grid built from LiDAR scans in the course frame, indexed [i, j] with i along x and j along y (as in CA2).
    Unknown cells count as free, so the planner is optimistic and the route is discovered as walls come into view.'''
    def __init__(self, bounds:tuple, resolution:float=GRID_RES):
        self.xmin, xmax, self.ymin, ymax = bounds
        self.res = resolution
        self.shape = (int(np.ceil((xmax - self.xmin) / resolution)), int(np.ceil((ymax - self.ymin) / resolution)))
        self.score = np.zeros(self.shape, dtype=np.float32) # > 0: seen occupied, < 0: seen free
        self.occupied = np.zeros(self.shape, dtype=bool)
        self.hit_sum = np.zeros((3,) + self.shape, dtype=np.float32) # sum of hit x, sum of hit y, number of hits per cell
        self.cx, self.cy = np.meshgrid(*self.centre(np.arange(self.shape[0]), np.arange(self.shape[1])), indexing="ij")
        r = int(np.ceil(SOFT_RADIUS / resolution))
        self.offsets = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1) if math.hypot(di, dj) * resolution <= SOFT_RADIUS + resolution]
        self.pad = r
        self.dirty = True
        self.refresh()

    def cell(self, x, y):
        '''World (x, y) -> cell (i, j). Works on scalars and arrays.'''
        return (np.floor((np.asarray(x) - self.xmin) / self.res).astype(int), np.floor((np.asarray(y) - self.ymin) / self.res).astype(int))

    def centre(self, i, j):
        '''Cell (i, j) -> world (x, y) of its centre. Works on scalars and arrays.'''
        return (self.xmin + (np.asarray(i) + 0.5) * self.res, self.ymin + (np.asarray(j) + 0.5) * self.res)

    def inside(self, i, j):
        return (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])

    def clip(self, cell:tuple) -> tuple:
        return (int(np.clip(cell[0], 0, self.shape[0] - 1)), int(np.clip(cell[1], 0, self.shape[1] - 1)))

    def _flat_cells(self, xy:np.ndarray) -> np.ndarray:
        i, j = self.cell(xy[:, 0], xy[:, 1])
        ok = self.inside(i, j)
        return np.unique(i[ok] * self.shape[1] + j[ok])

    def update(self, origin:np.ndarray, ends:np.ndarray, hits:np.ndarray, max_free:float=4.0):
        '''Add one scan. origin = LiDAR position, ends = ray end points (N, 2), hits = which ends are real returns.
        Cells the rays pass through lose evidence, so things that moved away disappear again; cells at real returns gain it.'''
        vec = ends - origin
        length = np.hypot(vec[:, 0], vec[:, 1])
        dirs = vec / np.maximum(length, 1e-9)[:, None]
        t = np.arange(0.0, max_free, 0.7 * self.res)
        keep = t[None, :] < (np.minimum(length, max_free) - 0.08)[:, None] # stop short of the end cell
        flat = self.score.reshape(-1)
        flat[self._flat_cells((origin + dirs[:, None, :] * t[None, :, None])[keep])] -= 1.0
        flat[self._flat_cells(ends[hits])] += 3.0
        np.clip(self.score, -4.0, 6.0, out=self.score)
        # Remember where in each cell the hits are, so distances are exact rather than cell centre to cell centre
        i, j = self.cell(ends[hits, 0], ends[hits, 1])
        ok = self.inside(i, j)
        idx = i[ok] * self.shape[1] + j[ok]
        sums = self.hit_sum.reshape(3, -1)
        for row, values in enumerate((ends[hits, 0][ok], ends[hits, 1][ok], np.ones(len(idx)))):
            np.add.at(sums[row], idx, values)
        full = sums[2] > 40.0 # keep the running mean, but let it follow if the thing moves
        sums[:, full] *= 0.5
        occupied = self.score >= 2.0
        if not np.array_equal(occupied, self.occupied):
            self.occupied = occupied
            self.dirty = True

    def refresh(self):
        '''Recompute (if the map changed) the distance from every cell centre to the nearest occupied cell's mean hit
        position (beyond SOFT_RADIUS: inf), the blocked mask (closer than BLOCK_RADIUS) and the A* step-cost multiplier.'''
        if not self.dirty:
            return
        H, W = self.shape
        p = self.pad
        n = np.maximum(self.hit_sum[2], 1.0)
        hx = np.where(self.hit_sum[2] > 0, self.hit_sum[0] / n, self.cx) # occupied cells without hits recorded: use the centre
        hy = np.where(self.hit_sum[2] > 0, self.hit_sum[1] / n, self.cy)
        occ, hx, hy = np.pad(self.occupied, p), np.pad(hx, p), np.pad(hy, p)
        dist = np.full(self.shape, np.inf, dtype=np.float32)
        for di, dj in self.offsets:
            window = (slice(p + di, p + di + H), slice(p + dj, p + dj + W))
            d = np.hypot(self.cx - hx[window], self.cy - hy[window])
            np.minimum(dist, np.where(occ[window], d, np.inf), out=dist)
        dist[dist > SOFT_RADIUS] = np.inf
        self.dist = dist
        self.blocked = dist < BLOCK_RADIUS
        self.cost = 1.0 + PROX_COST * np.clip((SOFT_RADIUS - dist) / (SOFT_RADIUS - BLOCK_RADIUS), 0.0, 1.0)
        self.dirty = False

    def line_of_sight(self, a, b, skip:float=0.1) -> bool:
        '''True if the straight line from a to b (world) crosses no blocked cell, ignoring its first `skip` metres
        (the robot itself may be standing in a blocked cell).'''
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length <= skip:
            return True
        s = np.linspace(skip / length, 1.0, int(length / (0.5 * self.res)) + 2)
        i, j = self.cell(a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]))
        ok = self.inside(i, j)
        return not np.any(self.blocked[i[ok], j[ok]])

def nearest_free(blocked:np.ndarray, cell:tuple, max_r:int=8):
    '''The unblocked cell nearest to `cell` within max_r cells, or None.'''
    i, j = cell
    H, W = blocked.shape
    i0, i1, j0, j1 = max(i - max_r, 0), min(i + max_r + 1, H), max(j - max_r, 0), min(j + max_r + 1, W)
    free = np.argwhere(~blocked[i0:i1, j0:j1])
    if len(free) == 0:
        return None
    k = int(np.argmin((free[:, 0] + i0 - i) ** 2 + (free[:, 1] + j0 - j) ** 2))
    return (int(free[k, 0] + i0), int(free[k, 1] + j0))

def astar(blocked:np.ndarray, cost:np.ndarray, start:tuple, goal:tuple) -> tuple:
    '''A* from CA2, reworked so it can replan every half second: heapq frontier instead of a linear search, integer
    cell ids, 8-connected moves (never cutting a corner past a blocked cell), octile heuristic, and each step's cost
    scaled by `cost` so paths keep away from walls. Returns (path as a list of (i, j) cells, whether it reaches the goal).
    If the goal can't be reached, the path leads to the reachable cell closest to it -- the goal only needs to be rough.'''
    H, W = blocked.shape
    blocked_flat = blocked.ravel().tolist()
    cost_flat = cost.ravel().tolist()
    gi, gj = goal
    def heuristic(i, j):
        di, dj = abs(i - gi), abs(j - gj)
        return max(di, dj) + (SQRT2 - 1.0) * min(di, dj)
    start_id, goal_id = start[0] * W + start[1], gi * W + gj
    best_id, best_h = start_id, heuristic(*start)
    g = {start_id: 0.0}
    parent = {start_id: None}
    closed = bytearray(H * W)
    frontier = [(best_h, 0.0, start_id)]
    while frontier:
        f, g_here, cid = heapq.heappop(frontier)
        if closed[cid]:
            continue
        closed[cid] = 1
        if f - g_here < best_h:
            best_id, best_h = cid, f - g_here
        if cid == goal_id:
            break
        i, j = divmod(cid, W)
        for di, dj, step in NEIGHBOURS:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= H or nj < 0 or nj >= W:
                continue
            nid = ni * W + nj
            if closed[nid] or blocked_flat[nid] or (di and dj and (blocked_flat[i * W + nj] or blocked_flat[ni * W + j])):
                continue
            g_new = g_here + step * cost_flat[nid]
            if g_new < g.get(nid, math.inf):
                g[nid] = g_new
                parent[nid] = cid
                heapq.heappush(frontier, (g_new + heuristic(ni, nj), g_new, nid))
    path, cid = [], best_id
    while cid is not None:
        path.append(divmod(cid, W))
        cid = parent[cid]
    path.reverse()
    return path, best_id == goal_id


class PlateTracker:
    '''Follows the swinging plate on each line across the corridor: lateral position over time, velocity, how far it
    swings and how fast it gets. expected_x = rough x of each line, used until that plate has been seen.'''
    def __init__(self, expected_x:list):
        self.lines = [{"x": x, "seen": False, "hist": deque(maxlen=100), "ymin": math.inf, "ymax": -math.inf, "v": 0.0,
                       "vpeak": 0.0, "dir": 0, "reversals": 0, "length": 0.3, "first": None, "last": None} for x in expected_x]

    def update(self, t:float, plates:list):
        '''Add the plate measurements (x, y, length) from one scan.'''
        for x, y, length in sorted(plates, key=lambda m: -m[2]): # longest (least hidden) first
            k = min(range(len(self.lines)), key=lambda n: abs(self.lines[n]["x"] - x))
            line = self.lines[k]
            if abs(line["x"] - x) > (0.12 if line["seen"] else 0.3) or line["last"] == t:
                continue
            line["x"] = 0.8 * line["x"] + 0.2 * x if line["seen"] else x
            line["seen"] = True
            line["first"] = t if line["first"] is None else line["first"]
            line["last"] = t
            line["hist"].append((t, y))
            line["ymin"], line["ymax"] = min(line["ymin"], y), max(line["ymax"], y)
            if length > line["length"]:
                line["length"] = min(length, 0.6)
            recent = np.array([s for s in line["hist"] if t - s[0] <= 0.5])
            if len(recent) >= 3 and recent[-1, 0] - recent[0, 0] > 0.15:
                v = float(np.clip(np.polyfit(recent[:, 0] - t, recent[:, 1], 1)[0], -3.0, 3.0))
                line["v"] = v
                line["vpeak"] = max(line["vpeak"], abs(v))
                if abs(v) > 0.05:
                    direction = 1 if v > 0 else -1
                    if line["dir"] and direction != line["dir"]:
                        line["reversals"] += 1
                    line["dir"] = direction

    def time_until_blocked(self, k:int, lane:float, half_band:float, now:float, speed_floor:float):
        '''Seconds until plate k could overlap a robot lane centred on y = lane (robot half-width + margin = half_band).
        0 = it does now (or we have not watched it long enough), inf = it never swings that far, None = never seen.'''
        line = self.lines[k]
        if not line["seen"] or now - line["last"] > 1.5: # never seen, or lost (hidden behind the other plate?)
            return None
        if len(line["hist"]) < 5 or now - line["first"] < 1.0:
            return 0.0
        y = line["hist"][-1][1] + line["v"] * min(now - line["last"], 0.3)
        w = half_band + line["length"] / 2
        gap = abs(y - lane) - w
        if gap <= 0:
            return 0.0
        if line["reversals"] >= 2 and (lane - w > line["ymax"] or lane + w < line["ymin"]):
            return math.inf # lane is outside the swing
        speed = max(line["vpeak"], abs(line["v"]), speed_floor if line["vpeak"] < 0.05 else 0.05)
        if line["v"] * (y - lane) > 0 and abs(line["v"]) > 0.05: # moving away: out to its far end, then back
            far = line["ymax"] if line["v"] > 0 else line["ymin"]
            return (2 * abs(far - y) + gap) / speed
        return gap / speed

class LidarNode(Node):
    def __init__(self):
        """Node constructor"""
        super().__init__("LidarNode")
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to lidar
        self.is_simulation = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360: 
            self.is_simulation = True
        else:
            self.is_simulation = False

class ObstacleCourseNode(Node):
    '''Node to navigate obstacle course, using pose from either gazebo odometer or optitrack and lidar scan data'''
    def __init__(self, is_simulation:bool=True, calibrate:bool=False):
        super().__init__('obstaclecourse')
        self.get_logger().info("Starting ObstacleCourseNode")

        self.is_simulation = is_simulation
        self.calibrate = calibrate # Real robot: only print poses, never drive
        self.raw_pose = None # Pose exactly as Optitrack reported it (real robot only)
        self.frame = self.course = None
        self._ticks = 0
        self._calib_ref = None # maze pose where the --calibrate heading check started
        self._pose_msgs = 0 # Optitrack messages since the last --calibrate print

        if is_simulation:
            self.max_translate_velocity = 1.4
        else:
            self.max_translate_velocity = 0.3 # Please keep this in place; 0.3m/s is more than fast enough 

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to LiDAR scan data

        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            # On the Bingda robots ROS_DOMAIN_ID is the robot's number, so the Optitrack topic is picked automatically
            domain = os.environ.get("ROS_DOMAIN_ID", "")
            self.robot_number = int(domain) if domain.isdigit() and 1 <= int(domain) <= 99 else 3
            self.frame, self.course = load_frame(self.robot_number)
            self.get_logger().info(f"Reading pose from /vrpn_mocap/bingda_{self.robot_number:03d}/pose")

            self.map_sub = self.create_subscription( 
                PoseStamped,
                f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose',
                self.optitrack_callback, 
                qos_profile
                )
            
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node       
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.pose = None
        self.last_scan = None

        self.init_other() # Final-project controller variables

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360:
            self.last_scan = np.array(msg.ranges)
        else:
            self.last_scan = np.array(msg.ranges)[::2] 

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def optitrack_callback(self, msg:PoseStamped):
        '''Callback to calculate 2D pose info from Optitrack node. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array((x,y,heading))
        self._pose_msgs += 1
        x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # CA2 maze frame, see optitrack_variables.config
        if self.pose is not None and math.hypot(x - self.pose[0], y - self.pose[1]) > 0.5: # further than the robot can drive between two poses
            self.pose_jump = (self.pose.copy(), np.array((x, y, heading)))
        self.pose = np.array((x,y,heading))
        return self.pose

    def to_course(self, pose=None):
        '''Convert a pose [x, y, heading] (default: self.pose) from the CA2 maze frame to the final-project course frame. Optional; see [course] in optitrack_variables.config'''
        pose = self.pose if pose is None else pose
        return np.array(maze_to_course(pose[0], pose[1], pose[2], self.course)) if self.course else np.array(pose)

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose 
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        self.pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -self.max_translate_velocity, self.max_translate_velocity)
        y = np.clip(y, -self.max_translate_velocity, self.max_translate_velocity)
        turn = np.clip(turn, -self.max_translate_velocity*2, self.max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        if self.pose is None:
            print("No pose detected")
            return # Does not run if no pose received from Odom or Optitrack

        if self.calibrate and self.raw_pose is not None:
            self._ticks += 1
            if self._ticks == 1:
                self.get_logger().info("HEADING CHECK: push the robot straight FORWARD by hand, ~0.4 m; it prints a result after every 0.3 m")
                # Course frame as INIT will set it: start the robot at the start point facing up the first corridor
                self._calib_start = {"origin_x": self.pose[0], "origin_y": self.pose[1], "rotation_deg": self.pose[2]}
                self.get_logger().info("COURSE: 'course (x, y)' is measured from where the robot is now (keep it at the start point, facing up the "
                                       "first corridor), the same frame as REAL_CONFIG's goal / bounds / door_region")
            # Heading check: push the robot straight FORWARD by hand; the direction it moved must match its heading
            if self._calib_ref is None:
                self._calib_ref = self.pose.copy()
            dx, dy = self.pose[0] - self._calib_ref[0], self.pose[1] - self._calib_ref[1]
            if self._ticks % 10 == 0: # twice a second
                ox, oy, oh = self.raw_pose
                rate, self._pose_msgs = self._pose_msgs / 0.5, 0
                status = " -- NO NEW POSES: Optitrack is not tracking this robot" if rate == 0 else f" | pushed {math.hypot(dx, dy):.2f} of 0.30 m"
                cx, cy, _ = maze_to_course(self.pose[0], self.pose[1], self.pose[2], self._calib_start)
                self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({self.pose[0]:+.3f}, {self.pose[1]:+.3f}, {self.pose[2]:+7.1f} deg)"
                                       f"  ->  course ({cx:+.2f}, {cy:+.2f}) | {rate:.0f} poses/s{status}")
            if math.hypot(dx, dy) > 0.3:
                moved = math.degrees(math.atan2(dy, dx))
                err = wrap_deg(self.pose[2] - moved)
                fix = f" -> WRONG: add  {self.robot_number} = {wrap_deg(self.frame['heading_offset_deg'] + err):.1f}  under [heading_offsets]"
                self.get_logger().info(f"HEADING CHECK: moved towards {moved:+.0f} deg, heading says {self.pose[2]:+.0f} deg, error {err:+.0f} deg" + (fix if abs(err) > 10.0 else " (OK)"))
                self._calib_ref = None
            return # Calibration mode: look, don't drive

        ###### INSERT CODE HERE ######
        self.run_mission()
        ###### INSERT CODE HERE ######

    # --- Final project controller ------------------------------------------------
    # Mission state machine (CA1 style): INIT -> NAVIGATE (A* on the live LiDAR map, CA2) -> DOOR_ALIGN -> DOOR_PASS
    # -> SWING_WAIT/SWING_CROSS for plate 1, then plate 2 -> DONE. RECOVER backs up and replans when stuck.
    def init_other(self):
        '''Final-project variables (CA2 pattern: called once from __init__).'''
        self.cfg = dict(SIM_CONFIG if self.is_simulation else REAL_CONFIG)
        self.holding = False # hold_heading is turning back towards course +x

        self.state, self.state_since = "INIT", 0.0
        self.init_ticks = 0
        self.axis_samples = [] # (wall axis, quality) from the first scans, real robot only
        # Course frame: sim = world frame; real robot = the [course] config if it is filled in, else the start pose lined up with the walls
        self.course_configured = (not self.is_simulation) and any(abs(v) > 1e-9 for v in self.course.values())
        self.frame_ready = self.is_simulation or self.course_configured
        self.start_frame = None
        self.pose_jump = None # (old, new) maze pose when Optitrack jumps further than the robot can drive (see handle_pose_jump)
        self.started_at = -math.inf # when INIT handed over to NAVIGATE
        self.cpose = None # pose in the course frame [x, y, heading]
        self.centre = None # centre of the footprint in the course frame [x, y]

        self.scan_obj = None # last scan array processed (the subscriber makes a new one per message)
        self.pts_r = None # latest valid LiDAR points, robot frame
        self.pts_c = None # same points, course frame
        self.sensor_c = None # LiDAR position, course frame
        self.grid = None
        self.mapping = True # stops once past the door, so the swinging plates never end up in the map

        self.path = None # A* path (cells) and the same as course-frame points
        self.path_xy = None
        self.path_idx = 0
        self.path_reached_goal = False
        self.path_goal = None
        self.last_plan = self.last_refresh = -math.inf
        self.replan_now = False
        self.prev_heading_err = None
        self.blocked_since = None
        self.progress_ref = None # (time, centre, heading) for stuck detection
        self.recoveries = 0
        self.resume_state = "NAVIGATE"
        self.recover_from = None # where RECOVER started backing up from
        self.narrow = False # the path just ahead squeezes past something (set by follow_path)

        self.door = None # smoothed door estimate {"x", "y", "width"}
        self.door_seen = -math.inf
        self.tracker = None
        self.walls_lr = [None, None] # y of the corridor walls left/right past the door
        self.end_x = None # x of the end wall
        self.swing_k = 0 # which plate line we are dealing with (0, 1)
        self.lane, self.lane_k = None, None
        self.wait_since = None
        self.rushing = False # crossing because the plate just swung past: no backing off

        self.pose_obj, self.pose_time = None, None
        self.last_log = self.last_warn = -math.inf
        self.replan_why = "" # what set replan_now (DEBUG prints)
        self.dbg_last = {} # DEBUG print throttles: key -> (time printed, prints skipped since)
        self.dbg_mode = None # how follow_path drove on the last tick
        self.dbg_err = self.dbg_room = None
        self.dbg_cmd = self.dbg_sent = (0.0, 0.0, 0.0) # the state's command, and what the safety layer let through
        self.last_map = -math.inf
        self.self_hits = 0 # LiDAR returns on the robot itself in the last scan (dropped)
        self.dbg_cmd_dist, self.dbg_pose_ref, self.dbg_pose_t = 0.0, None, -math.inf # DBG POSE: commanded distance and pose since the last print
        self.done_logged = False

    def clock(self) -> float:
        return time.monotonic()

    def run_mission(self):
        '''One tick: sense, update the map, step the mission state machine, check the command is safe, act.'''
        now = self.clock()
        if self.last_scan is None:
            self.move_2D()
            return
        if not self.is_simulation and self.pose_lost(now):
            self.move_2D()
            return
        if self.pose_jump is not None:
            self.handle_pose_jump(now)
        if self.frame_ready:
            self.update_course_pose()
        if self.last_scan is not self.scan_obj:
            self.scan_obj = self.last_scan
            self.process_scan(now)
        if LIDAR_CHECK:
            self.move_2D()
            if now - self.last_log >= 0.5:
                self.last_log = now
                k = int(np.argmin(np.hypot(self.pts_r[:, 0] - LIDAR_X, self.pts_r[:, 1])))
                bearing = math.degrees(math.atan2(self.pts_r[k, 1], self.pts_r[k, 0] - LIDAR_X))
                rel = self.pts_r - np.array((LIDAR_X, 0.0))
                ang, dist = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])), np.hypot(rel[:, 0], rel[:, 1])
                sides = []
                for name, a in (("front", 0.0), ("left", 90.0), ("back", 180.0), ("right", -90.0)):
                    near = np.abs((ang - a + 180.0) % 360.0 - 180.0) < 5.0
                    sides.append(f"{name} {dist[near].min():.2f}" if np.any(near) else f"{name} -")
                self.get_logger().info(f"LIDAR_CHECK: closest return at {bearing:+.0f} deg, {dist[k]:.2f} m (0 = straight ahead, +ve = left)"
                                       f" | nearest (m): {', '.join(sides)}")
            return

        cmd = (0.0, 0.0, 0.0)
        for _ in range(4): # a state that hands over returns None, so the next state runs in the same tick (like CA1)
            state = self.state
            result = getattr(self, "state_" + state.lower())(now)
            if result is not None:
                cmd = result
                break
            if self.state == state:
                break
        sent = self.safety_filter(*cmd, now)
        self.dbg_cmd, self.dbg_sent = cmd, sent
        self.dbg_cmd_dist += math.hypot(sent[0], sent[1]) * 0.05 # how far the commands should have moved it (20 Hz ticks)
        self.move_2D(*sent)
        self.log_status(now)

    # --- Sense --------------------------------------------------------------------
    def update_course_pose(self):
        '''self.pose (Gazebo world / CA2 maze frame) -> self.cpose and self.centre in the course frame.'''
        if self.is_simulation:
            x, y, h = self.pose
        elif self.start_frame is None:
            x, y, h = self.to_course()
        else:
            ox, oy, oh = self.start_frame
            x, y, h = maze_to_course(self.pose[0], self.pose[1], self.pose[2], {"origin_x": ox, "origin_y": oy, "rotation_deg": oh})
        self.cpose = np.array((x, y, h))
        hr = math.radians(h)
        self.centre = np.array((x + CENTRE_X * math.cos(hr), y + CENTRE_X * math.sin(hr)))

    def process_scan(self, now:float):
        '''New scan: points in the robot and course frames, map update, plate tracking.'''
        step = (self.cfg["lidar_step_deg"] or 360.0 / len(self.last_scan)) * (1.0 if self.cfg["lidar_ccw"] else -1.0)
        points, valid, dirs, ranges = lidar_points(self.last_scan, self.cfg["lidar_start_deg"], step)
        own = valid & (points[:, 0] > -ROBOT_REAR - SELF_MARGIN) & (points[:, 0] < ROBOT_FRONT + SELF_MARGIN) & \
              (np.abs(points[:, 1]) < ROBOT_HALF_WIDTH + SELF_MARGIN)
        self.self_hits = int(own.sum())
        valid = valid & ~own # the LiDAR seeing the robot itself: not an obstacle (it would be mapped on top of the robot), and no free-space info
        self.pts_r = points[valid]
        if not self.frame_ready:
            if self.state == "INIT":
                self.axis_samples.append(estimate_wall_axis(self.pts_r))
            return
        self.pts_c = to_frame(self.pts_r, self.cpose)
        self.sensor_c = to_frame(np.array([[LIDAR_X, 0.0]]), self.cpose)[0]
        if self.mapping and self.grid is not None:
            no_return = ~valid & (~np.isfinite(ranges) | (ranges >= 8.0)) # clear space up to 2 m along rays that hit nothing
            use = valid | no_return
            ends = np.array((LIDAR_X, 0.0)) + dirs * np.where(valid, ranges, 2.0)[:, None]
            self.grid.update(self.sensor_c, to_frame(ends[use], self.cpose), valid[use])
        if self.tracker is not None:
            self.update_tracker(now)

    def update_tracker(self, now:float):
        cx, cy = self.centre
        walls = side_walls(self.pts_c, cx, cy)
        door_lo, door_hi = self.door["y"] - self.door["width"] / 2, self.door["y"] + self.door["width"] / 2
        if walls[0] is not None and walls[0] < door_hi: # the corridor is at least as wide as the door
            walls[0] = None
        if walls[1] is not None and walls[1] > door_lo:
            walls[1] = None
        for n in range(2):
            if walls[n] is not None:
                self.walls_lr[n] = walls[n] if self.walls_lr[n] is None else 0.8 * self.walls_lr[n] + 0.2 * walls[n]
        y_left, y_right = self.walls_lr
        y_hi = y_left - 0.04 if y_left is not None else cy + 1.2
        y_lo = y_right + 0.04 if y_right is not None else cy - 1.2
        x_min = self.door["x"] + 0.12
        x_max = self.end_x - 0.12 if self.end_x is not None else cx + 2.5
        plates, walls_x = plate_measurements(self.pts_c, x_min, x_max, y_lo, y_hi)
        self.tracker.update(now, plates)
        beyond = [x for x in walls_x if x > self.tracker.lines[-1]["x"] + 0.2]
        if beyond:
            self.end_x = min(beyond) if self.end_x is None else 0.8 * self.end_x + 0.2 * min(beyond)

    def pose_lost(self, now:float) -> bool:
        '''Real robot: Optitrack always jitters, so a pose that stops changing for 0.5 s means tracking is lost. Compare
        values, not messages: Motive keeps streaming the last pose of a rigid body it can no longer see.'''
        if self.pose_obj is None or not np.array_equal(self.raw_pose, self.pose_obj):
            self.pose_obj, self.pose_time = self.raw_pose, now
        if now - self.pose_time > 0.5:
            self.warn_throttled(now, f"Optitrack pose unchanged for {now - self.pose_time:.1f} s (tracking lost?) -- stopped")
            return True
        return False

    # --- Act ----------------------------------------------------------------------
    def heading_control(self, err_deg:float, in_place:bool=False) -> float:
        '''Heading PD (CA2 turn/steer PID, in rad/s per rad). In place, never below min_turn so friction can't stall it.'''
        e = math.radians(err_deg)
        de = 0.0 if self.prev_heading_err is None else (e - self.prev_heading_err) / 0.05
        self.prev_heading_err = e
        turn = float(np.clip(self.cfg["turn_kp"] * e + self.cfg["turn_kd"] * de, -self.cfg["max_turn"], self.cfg["max_turn"]))
        if in_place and abs(err_deg) > 2.0 and abs(turn) < self.cfg["min_turn"]:
            turn = math.copysign(self.cfg["min_turn"], e)
        return turn

    def min_speed(self, v:float, err:float, tol:float) -> float:
        '''Real robot: commands below min_run don't move it, so bump them up while the error is still outside tolerance.'''
        if abs(err) > tol and 0.0 < abs(v) < self.cfg["min_run"]:
            return math.copysign(self.cfg["min_run"], v)
        return v

    def safety_filter(self, vx:float, vy:float, turn:float, now:float) -> tuple:
        '''Always on: limit the speed so the robot can stop (0.35 s at that speed + 4 cm) before its footprint, slid along
        the commanded direction, would touch a LiDAR return. Too close = stop translating and ask for a new plan.
        Turning on the spot is checked too: the front corners swing out 0.23 m from the rear axle.'''
        if self.pts_r is None:
            return vx, vy, turn
        if abs(turn) > 0.05 and not rotation_clear(self.pts_r, math.copysign(max(8.0, math.degrees(abs(turn)) * 0.3), turn)):
            self.replan_now, self.replan_why = True, "safety: no room to turn"
            self.warn_throttled(now, f"[{self.state}] safety: no room to turn that way")
            self.debug("safety_turn", f"SAFETY no room to turn {'left' if turn > 0 else 'right'} (turn {turn:+.2f}); nearest {self.fmt_nearest()}", 0.5)
            turn = 0.0
        speed = math.hypot(vx, vy)
        if speed < 1e-3:
            return vx, vy, turn
        margin = 0.005 if self.state.startswith("DOOR") else 0.01 if self.narrow and self.state == "NAVIGATE" else 0.02
        free = free_distance(self.pts_r, vx / speed, vy / speed, margin)
        allowed = (free - 0.04) / 0.35
        if allowed >= speed:
            self.blocked_since = None
            return vx, vy, turn
        if allowed > 0.03:
            self.debug("safety_slow", f"SAFETY slowing {speed:.2f} -> {allowed:.2f} m/s: {free:.2f} m free that way; nearest {self.fmt_nearest()}", 1.0)
            return vx * allowed / speed, vy * allowed / speed, turn
        if self.blocked_since is None:
            self.blocked_since = now
        self.replan_now, self.replan_why = True, "safety stop"
        self.warn_throttled(now, f"[{self.state}] safety stop: something in the way")
        self.debug("safety_stop", f"SAFETY STOP: cmd vx {vx:+.2f} vy {vy:+.2f} has only {free:.2f} m free that way; nearest {self.fmt_nearest()}", 0.5)
        return 0.0, 0.0, turn

    def hold_heading(self) -> float:
        '''The robot never turns to steer. Only if its heading drifts more than heading_hold_deg off course +x does it turn
        slowly back, until within 2 deg, so the footprint stays lined up with the walls. heading_hold_deg None = never turn.'''
        limit = self.cfg["heading_hold_deg"]
        eh = wrap_deg(-self.cpose[2])
        if limit is None:
            return 0.0
        if abs(eh) > limit:
            self.holding = True
        elif abs(eh) < 2.0:
            self.holding = False
        if not self.holding:
            self.prev_heading_err = None
            return 0.0
        return self.heading_control(eh, True)

    def slide(self, vx:float, vy:float) -> tuple:
        '''Course-frame velocity -> (vx, vy, turn) command. Rotated into the robot frame with the current heading, so the
        robot moves the right way even if it has turned a little; the only turning is hold_heading.'''
        hr = math.radians(self.cpose[2])
        c, s = math.cos(hr), math.sin(hr)
        return (c * vx + s * vy, -s * vx + c * vy, self.hold_heading())

    def drive_to(self, target, speed:float, tol=(0.04, 0.03)) -> tuple:
        '''Slide the footprint centre to target (course frame) without turning: along course x at up to `speed`, along y
        at up to strafe_speed. Returns (command, arrived).'''
        c = self.centre
        ex, ey = target[0] - c[0], target[1] - c[1]
        if abs(ex) < tol[0] and abs(ey) < tol[1]:
            return (0.0, 0.0, 0.0), True
        vx = self.min_speed(float(np.clip(2.0 * ex, -speed, speed)), ex, tol[0])
        vy = self.min_speed(float(np.clip(2.0 * ey, -self.cfg["strafe_speed"], self.cfg["strafe_speed"])), ey, tol[1])
        return self.slide(vx, vy), False

    def path_lookahead(self) -> tuple:
        '''Move along the path and pick the point to head for: the furthest point of the path within the look-ahead that
        is in line of sight. Returns (target, distance to the end of the path); target is None once at the end.'''
        c = self.centre
        pts = self.path_xy
        d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
        self.path_idx += int(np.argmin(d[self.path_idx:self.path_idx + 20]))
        if self.path_idx >= len(pts) - 1 and d[-1] < 0.06:
            return None, d[-1]
        # Narrow stretch coming up (path within 0.2 m of something)? Then look less far ahead, so the robot enters gaps
        # straight instead of cutting the corner into them.
        ahead = np.array(self.path[self.path_idx:self.path_idx + 10])
        self.narrow = bool(np.any(self.grid.dist[ahead[:, 0], ahead[:, 1]] < 0.2))
        lookahead = 0.15 if self.narrow else self.cfg["lookahead"]
        far = self.path_idx
        while far + 1 < len(pts) and d[far + 1] <= lookahead:
            far += 1
        while far > self.path_idx + 1 and not self.grid.line_of_sight(c, pts[far]):
            far -= 1
        return pts[max(far, min(self.path_idx + 1, len(pts) - 1))], d[-1]

    def follow_path(self, speed:float) -> tuple:
        '''Slide along the path towards the look-ahead point without turning: forwards, backwards, sideways or diagonally.
        Slower in tight spaces and for the end of the path; the sideways part stays within strafe_speed.
        Returns (command, arrived at the end).'''
        target, left = self.path_lookahead()
        if target is None:
            return (0.0, 0.0, 0.0), True
        c = self.centre
        ex, ey = target[0] - c[0], target[1] - c[1]
        dist = max(math.hypot(ex, ey), 1e-6)
        i, j = self.grid.cell(c[0], c[1])
        room = float(self.grid.dist[i, j]) if self.grid.inside(i, j) else math.inf # distance to the nearest obstacle
        v = speed * float(np.clip((room - 0.1) / 0.25, 0.35, 1.0)) # tight spaces
        v = min(v, 0.05 + 1.5 * left) # end of the path
        vx, vy, turn = self.slide(v * ex / dist, v * ey / dist)
        if abs(vy) > self.cfg["strafe_speed"]: # sideways part too fast: slow the whole move down, keeping its direction
            vx, vy = vx * self.cfg["strafe_speed"] / abs(vy), math.copysign(self.cfg["strafe_speed"], vy)
        v = math.hypot(vx, vy)
        if v > 1e-6:
            k = self.min_speed(v, left, 0.06) / v
            vx, vy = vx * k, vy * k
        direction = math.degrees(math.atan2(vy, vx)) # robot frame: 0 = forwards, +90 = left
        self.debug_mode("forward" if abs(direction) <= 45.0 else "backward" if abs(direction) >= 135.0 else "left" if direction > 0 else "right",
                        direction, room)
        return (vx, vy, turn), False

    # --- Plan -----------------------------------------------------------------------
    def plan(self, now:float, goal=None):
        '''A* from the robot to goal (default: the rough course goal) on the current map.'''
        self.grid.refresh()
        goal = self.cfg["goal"] if goal is None else goal
        start = nearest_free(self.grid.blocked, self.grid.clip(self.grid.cell(*self.centre)))
        forced = self.replan_now or self.path is None or self.path_goal != goal
        why = "no path yet" if self.path is None else "new goal" if self.path_goal != goal else (self.replan_why or "replan requested") if self.replan_now else ""
        self.last_plan, self.replan_now, self.path_goal = now, False, goal
        if start is None:
            self.path = None
            self.warn_throttled(now, "No free cell near the robot to plan from")
            return
        path, reached = astar(self.grid.blocked, self.grid.cost, start, self.grid.clip(self.grid.cell(*goal)))
        if not forced and path[-1] == self.path[-1] and not self.path_blocked() and \
                self.path_cost(self.path[self.path_idx:]) <= 1.2 * self.path_cost(path) + 2.0:
            return # routine replan found nothing clearly better: keep the current path, so the robot doesn't dither between two gaps
        old = 0 if self.path is None else len(self.path) - self.path_idx
        if not why:
            why = "path end moved" if path[-1] != self.path[-1] else "old path blocked" if self.path_blocked() else "found a clearly cheaper path"
        self.path, self.path_reached_goal = path, reached
        cells = np.array(path)
        self.path_xy = np.column_stack(self.grid.centre(cells[:, 0], cells[:, 1]))
        self.path_idx = 0
        b = self.path_bearing()
        self.debug("plan", f"REPLAN ({why}): {old} -> {len(path)} cells, {'reaches goal' if reached else 'to closest reachable cell'}, "
                   f"first 0.4 m heads {b:+.0f} deg from heading" + (" = BEHIND the robot" if abs(b) > 120.0 else ""), 0.3)

    def path_cost(self, cells:list) -> float:
        '''A* cost of a list of cells on the current map (in cells).'''
        if len(cells) < 2:
            return 0.0
        a = np.array(cells)
        steps = np.where(np.abs(np.diff(a, axis=0)).sum(axis=1) == 2, SQRT2, 1.0)
        return float((steps * self.grid.cost[a[1:, 0], a[1:, 1]]).sum())

    def path_blocked(self) -> bool:
        cells = np.array(self.path[self.path_idx + 2:])
        return len(cells) > 0 and bool(np.any(self.grid.blocked[cells[:, 0], cells[:, 1]]))

    def keep_path_fresh(self, now:float, goal=None):
        '''Replan on a timer, when the map shows the path is now blocked, or when the safety layer stopped us.'''
        if now - self.last_refresh > self.cfg["refresh_period"]:
            self.grid.refresh()
            self.last_refresh = now
            if self.path is not None and self.path_blocked():
                self.replan_now, self.replan_why = True, "old path now blocked on the map"
        goal_moved = goal is not None and (self.path_goal is None or math.hypot(goal[0] - self.path_goal[0], goal[1] - self.path_goal[1]) > 0.1)
        if self.path is None or self.replan_now or goal_moved or now - self.last_plan > self.cfg["replan_period"]:
            self.plan(now, goal)

    def is_stuck(self, now:float) -> bool:
        c, h = self.centre, self.cpose[2]
        if self.progress_ref is None or math.hypot(*(c - self.progress_ref[1])) > 0.05 or abs(wrap_deg(h - self.progress_ref[2])) > 15.0:
            self.progress_ref = (now, c.copy(), h)
            return False
        return now - self.progress_ref[0] > 3.0

    # --- Door -----------------------------------------------------------------------
    def update_door(self, now:float, ahead=(0.0, 2.0)):
        '''Look for the door again and blend it into self.door. Returns the raw detection (or None).'''
        cx, cy = self.centre
        found = find_door(self.pts_c, self.sensor_c, cx, cy, self.cfg["door_region"], ahead=ahead, half_window=1.5,
                          width=self.cfg["door_width"], support=self.cfg["door_support"])
        if found is None:
            return None
        if self.door is None or now - self.door_seen > 1.0:
            self.door = dict(found)
        elif abs(found["x"] - self.door["x"]) < 0.15 and abs(found["y"] - self.door["y"]) < 0.2:
            for key in ("x", "y", "width"):
                self.door[key] = 0.7 * self.door[key] + 0.3 * found[key]
        else:
            return None # a different gap; keep the current estimate
        self.door_seen = now
        return found

    def start_swing(self, now:float):
        '''Past the door: stop mapping, start (or keep) tracking the plates, deal with plate 1.'''
        self.mapping = False
        if self.tracker is None:
            spacing = self.cfg["plate_spacing"]
            self.tracker = PlateTracker([self.door["x"] + spacing, self.door["x"] + 2 * spacing])
        self.swing_k = 0
        self.lane = None
        self.set_state("SWING_WAIT", now, "past the door")

    # --- Swinging plates ------------------------------------------------------------
    def plate_x(self, k:int) -> float:
        return self.tracker.lines[k]["x"]

    def clear_x(self, k:int) -> float:
        '''Centre x at which the whole robot is past plate line k (k = -1: past the door).'''
        line_back = self.door["x"] if k < 0 else self.plate_x(k) + PLATE_HALF_T
        return line_back + 0.04 + HALF_LENGTH

    def stage_x(self, k:int) -> float:
        '''Centre x to wait at before plate line k: front bumper 0.1 m short of the plate.'''
        return max(self.plate_x(k) - PLATE_HALF_T - 0.10 - HALF_LENGTH, self.clear_x(k - 1))

    def cross_target(self, k:int) -> float:
        if k == 0:
            return self.stage_x(1)
        target = self.clear_x(1) + 0.08
        if self.end_x is not None:
            target = min(target, self.end_x - 0.12 - HALF_LENGTH)
        return max(target, self.clear_x(1))

    def lane_band(self) -> float:
        return ROBOT_HALF_WIDTH + 0.06

    def pick_lane(self, k:int) -> float:
        '''Once plate k has been seen swinging both ways, pick the lane it blocks least (towards or beyond one end of its
        swing, so it is out of the way for one long stretch per swing rather than two short ones).'''
        cy = self.centre[1]
        if self.lane_k == k and self.lane is not None:
            return self.lane
        line = self.tracker.lines[k]
        y_left, y_right = self.walls_lr
        if line["reversals"] < 2 or y_left is None or y_right is None:
            return cy
        lo, hi = y_right + ROBOT_HALF_WIDTH + 0.07, y_left - ROBOT_HALF_WIDTH - 0.07
        if lo >= hi:
            return cy
        lanes = np.arange(lo, hi + 1e-6, 0.02)
        w = self.lane_band() + line["length"] / 2
        overlap = np.clip(np.minimum(lanes + w, line["ymax"]) - np.maximum(lanes - w, line["ymin"]), 0.0, None)
        score = overlap / max(line["ymax"] - line["ymin"], 0.05) + 0.1 * np.abs(lanes - cy)
        self.lane, self.lane_k = float(lanes[int(np.argmin(score))]), k
        self.get_logger().info(f"Plate {k + 1} swings y {line['ymin']:.2f}..{line['ymax']:.2f}, peak {line['vpeak']:.2f} m/s: using lane y={self.lane:.2f}")
        return self.lane

    def time_free(self, k:int, now:float):
        return self.tracker.time_until_blocked(k, self.centre[1], self.lane_band(), now, self.cfg["plate_speed_floor"])

    def lane_clear(self, k:int) -> bool:
        '''Nothing in the robot's lane between its front and just past plate line k.'''
        return swept_clear(self.pts_r, self.plate_x(k) + PLATE_HALF_T + 0.05 - (self.centre[0] + HALF_LENGTH) + 0.05, 0.0, 0.02)

    # --- Mission states -------------------------------------------------------------
    def state_init(self, now:float):
        '''Wait for pose and LiDAR to settle (like CA2's delay_counter), then fix the course frame and start.'''
        self.init_ticks += 1
        if self.pts_r is None or self.init_ticks < self.cfg["settle_ticks"]:
            return (0.0, 0.0, 0.0)
        if not self.frame_ready:
            good = [a for a, quality in self.axis_samples if quality > 0.3]
            axis = float(np.median(good)) if good else 0.0
            if abs(axis) > 20.0:
                self.get_logger().warn(f"Walls look {axis:.0f} deg off the robot's heading; assuming the robot faces up the first corridor")
                axis = 0.0
            self.start_frame = (self.pose[0], self.pose[1], wrap_deg(self.pose[2] + axis))
            self.frame_ready = True
            self.get_logger().info(f"Course frame: origin ({self.pose[0]:.2f}, {self.pose[1]:.2f}), +x at {self.start_frame[2]:.1f} deg (walls {axis:+.1f} deg from the robot's heading)")
            self.update_course_pose()
        self.grid = OccupancyGrid(self.cfg["bounds"])
        self.process_scan(now)
        if self.is_simulation and self.cpose[0] > 3.2: # Gazebo started past the gate (see gz_fp.sh): test the swinging plates only
            self.door = {"x": self.cpose[0] - 0.3, "y": float(self.cpose[1]), "width": 0.4}
            self.start_swing(now)
            return None
        self.started_at = now
        self.set_state("NAVIGATE", now, "start")
        return None

    def handle_pose_jump(self, now:float):
        '''Optitrack moved the robot further than it can drive (rigid body lost and found again somewhere else). Just after
        the start, the course frame was probably fixed on the bad pose: fix it again from the new one.'''
        old, new = self.pose_jump
        self.pose_jump = None
        self.get_logger().warn(f"Optitrack pose JUMPED {math.hypot(new[0] - old[0], new[1] - old[1]):.2f} m: maze ({old[0]:.2f}, {old[1]:.2f}, {old[2]:.0f} deg)"
                               f" -> ({new[0]:.2f}, {new[1]:.2f}, {new[2]:.0f} deg). Check robot {self.robot_number}'s rigid body in Motive")
        if self.state == "INIT" or (not self.course_configured and now - self.started_at < 5.0):
            self.start_frame, self.frame_ready = None, self.course_configured
            self.grid, self.path, self.axis_samples, self.init_ticks = None, None, [], 0
            self.set_state("INIT", now, "pose jumped just after the start, fixing the course frame again")

    def state_navigate(self, now:float):
        '''Zig-zag section: A* on the live LiDAR map towards the rough goal, replanned all the time.'''
        if abs(wrap_deg(self.cpose[2])) < 35.0: # door ahead?
            cx, cy = self.centre
            why = [] if DEBUG else None
            door = find_door(self.pts_c, self.sensor_c, cx, cy, self.cfg["door_region"],
                             width=self.cfg["door_width"], support=self.cfg["door_support"], why=why)
            if door is not None and door["x"] - (cx + HALF_LENGTH) < 1.2 and abs(door["y"] - cy) < 0.8:
                self.door, self.door_seen = door, now
                self.set_state("DOOR_ALIGN", now, f"door at x={door['x']:.2f}, y={door['y']:.2f}, {door['width']:.2f} m wide, "
                               f"wall {door['runs'][0]:.2f} / {door['runs'][1]:.2f} m either side, tilted {door['tilt']:+.0f} deg")
                return None
            if door is not None and why is not None:
                why.insert(0, f"door x={door['x']:.2f} y={door['y']:.2f} seen but not within 1.2 m ahead and 0.8 m to the side")
            if why:
                self.debug("door", "DOOR not taken: " + " | ".join(why[:4]) + (f" (+{len(why) - 4} more)" if len(why) > 4 else ""), 1.0)
        self.keep_path_fresh(now)
        if self.path is None:
            return (0.0, 0.0, 0.0)
        cmd, arrived = self.follow_path(self.cfg["cruise_speed"])
        if arrived:
            self.set_state("DOOR_ALIGN", now, "reached the goal" if self.path_reached_goal else "end of the reachable area, door should be ahead")
            return None
        if self.is_stuck(now):
            self.resume_state = "NAVIGATE"
            self.set_state("RECOVER", now, "no progress")
            return None
        return cmd

    def state_recover(self, now:float):
        '''Back up ~0.35 m (the safety layer checks behind), so there is room to line up again, then plan again -- like
        CA1's BLOCKED state.'''
        if self.recover_from is None:
            self.recover_from = self.centre.copy()
        if math.hypot(*(self.centre - self.recover_from)) < 0.35 and now - self.state_since < 4.0:
            return (-self.cfg["backup_speed"], 0.0, self.hold_heading())
        self.recover_from = None
        self.recoveries += 1
        self.replan_now, self.replan_why = True, "after recovery"
        self.progress_ref = None
        self.set_state(self.resume_state, now, f"recovery {self.recoveries}")
        return None

    def state_door_align(self, now:float):
        '''Get right up to the door first, then line up there: centred on the gap, front bumper door_stage_gap short of the
        wall. Lining up close gives the best view of the gap and only a short straight run in afterwards.'''
        self.update_door(now)
        if self.door is None:
            if now - self.state_since > 4.0:
                self.resume_state = "NAVIGATE"
                self.set_state("RECOVER", now, "no door found")
                return None
            return (0.0, 0.0, self.hold_heading()) # keep looking
        target = (self.door["x"] - HALF_LENGTH - self.cfg["door_stage_gap"], self.door["y"])
        if math.hypot(target[0] - self.centre[0], target[1] - self.centre[1]) > 0.08: # not there yet: A* path all the way, no fine lining up
            self.keep_path_fresh(now, target)
            if self.path is not None:
                cmd, arrived = self.follow_path(self.cfg["door_speed"])
                if not arrived:
                    return cmd
        cmd, arrived = self.drive_to(target, self.cfg["door_speed"], tol=(0.04, 0.02))
        if arrived:
            self.set_state("DOOR_PASS", now, f"lined up on y={self.door['y']:.2f}")
            return None
        if now - self.state_since > 25.0:
            self.resume_state = "NAVIGATE"
            self.door = None
            self.set_state("RECOVER", now, "could not line up with the door")
            return None
        return cmd

    def state_door_pass(self, now:float):
        '''Through the gap, slowly and straight, staying centred on it.'''
        if self.tracker is None: # start watching the plates through the door
            spacing = self.cfg["plate_spacing"]
            self.tracker = PlateTracker([self.door["x"] + spacing, self.door["x"] + 2 * spacing])
        c = self.centre
        if c[0] + HALF_LENGTH < self.door["x"] - 0.05:
            self.update_door(now, ahead=(0.0, 1.0))
        if c[0] - HALF_LENGTH > self.door["x"] + 0.04:
            self.start_swing(now)
            return None
        ey = self.door["y"] - c[1]
        before = c[0] + HALF_LENGTH < self.door["x"] - 0.02
        vx = 0.0 if before and abs(ey) > 0.04 else self.cfg["door_speed"] # off the gap: side-step back onto it before going in
        vy = float(np.clip(2.5 * ey, -0.6 * self.cfg["strafe_speed"], 0.6 * self.cfg["strafe_speed"]))
        if not before: # inside the door: keep off whichever side is closer
            left, right = side_clearance(self.pts_r)
            if min(left, right) < 0.04:
                vy = 0.03 if left > right else -0.03
        return self.slide(vx, vy)

    def state_swing_wait(self, now:float):
        '''Hold just before plate line k, watch the plate, move to the best lane, and go as soon as it swings past the lane
        (or once it can't reach the lane in time).'''
        k = self.swing_k
        lane = self.pick_lane(k)
        cmd, ready = self.drive_to((self.stage_x(k), lane), self.cfg["door_speed"], tol=(0.05, 0.03))
        if not ready:
            return cmd
        if self.wait_since is None:
            self.wait_since = now
        t_cross = (self.clear_x(k) - self.centre[0]) / self.cfg["dash_speed"] + self.cfg["cross_latency"]
        t_free = self.time_free(k, now)
        if t_free is None: # never seen (maybe not there) or lost (hidden?): go only once the lane has stayed clear for a while
            line = self.tracker.lines[k]
            unseen_for = now - line["last"] if line["seen"] else now - self.wait_since
            timeout = self.cfg["unseen_timeout"] * (2.0 if line["seen"] else 1.0)
            go = unseen_for > timeout and now - self.wait_since > self.cfg["unseen_timeout"] and self.lane_clear(k)
        else:
            go = t_free > self.cfg["cross_safety"] * t_cross
        self.rushing = not go and self.plate_just_passed(k, now)
        if self.rushing:
            self.set_state("SWING_CROSS", now, f"plate {k + 1} just swung past the lane, rushing")
            return None
        if go:
            self.set_state("SWING_CROSS", now, f"plate {k + 1} " + ("not seen" if t_free is None else f"needs {t_free:.1f} s, crossing takes {t_cross:.1f} s"))
            return None
        return cmd

    def plate_just_passed(self, k:int, now:float) -> bool:
        '''Plate k overlapped our lane within the last 0.5 s and is now clear of it, moving away: it has to swing out to
        its far end and back before it can block us again, the longest gap it ever leaves.'''
        line = self.tracker.lines[k]
        if not line["seen"] or now - line["last"] > 0.3 or abs(line["v"]) < 0.05:
            return False
        lane, w = self.centre[1], self.lane_band() + line["length"] / 2
        t, y = line["hist"][-1]
        if abs(y - lane) <= w or line["v"] * (y - lane) <= 0:
            return False
        return any(abs(y_old - lane) <= w for t_old, y_old in line["hist"] if t - t_old <= 0.5)

    def state_swing_cross(self, now:float):
        '''Dash past plate line k. Until the front reaches the plate, back off if it will reach our lane first -- unless
        we went because it had just swung past, then commit.'''
        k = self.swing_k
        c = self.centre
        target = self.cross_target(k)
        if c[0] >= target - 0.03:
            if k == 0:
                self.swing_k, self.lane = 1, None
                self.set_state("SWING_WAIT", now, "past plate 1")
            else:
                self.set_state("DONE", now, "past plate 2")
            return None
        if not self.rushing and c[0] + HALF_LENGTH < self.plate_x(k) - PLATE_HALF_T - 0.02:
            t_free = self.time_free(k, now)
            remaining = (self.clear_x(k) - c[0]) / self.cfg["dash_speed"] + 0.5 * self.cfg["cross_latency"]
            if t_free is not None and t_free < remaining:
                self.set_state("SWING_WAIT", now, f"plate {k + 1} coming, backing off")
                return None
        vx = self.cfg["dash_speed"] if c[0] < self.clear_x(k) else min(self.cfg["dash_speed"], 0.08 + 2.0 * (target - c[0]))
        return self.slide(vx, 0.0)

    def state_done(self, now:float):
        if not self.done_logged:
            self.done_logged = True
            self.get_logger().info(f"Past the swinging obstacles at {self.fmt_pose()} -- done")
        return (0.0, 0.0, 0.0)

    # --- Logging ---------------------------------------------------------------------
    def set_state(self, state:str, now:float, why:str=""):
        self.get_logger().info(f"{self.state} -> {state}" + (f" ({why})" if why else "") + f" at {self.fmt_pose()}")
        self.state, self.state_since = state, now
        self.prev_heading_err = None
        self.wait_since = None
        self.progress_ref = None
        self.dbg_mode = self.dbg_err = self.dbg_room = None
        if state == "RECOVER":
            self.print_map("entering RECOVER")

    def fmt_pose(self) -> str:
        p = self.cpose if self.cpose is not None else self.pose
        return f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.0f} deg)"

    def warn_throttled(self, now:float, msg:str):
        if now - self.last_warn >= 1.0:
            self.last_warn = now
            self.get_logger().warn(msg)

    def debug(self, key:str, msg:str, period:float=0.0):
        '''DEBUG print, at most once every `period` s per key; says how many it skipped in between.'''
        if not DEBUG:
            return
        now = self.clock()
        last, skipped = self.dbg_last.get(key, (-math.inf, 0))
        if now - last < period:
            self.dbg_last[key] = (last, skipped + 1)
            return
        self.dbg_last[key] = (now, 0)
        self.get_logger().info(f"DBG {msg}" + (f"  [+{skipped} skipped]" if skipped else ""))

    def debug_mode(self, mode:str, err:float, room:float):
        '''DEBUG: remember how follow_path is driving; print when that changes (this is what makes it go back and forth).'''
        self.dbg_err, self.dbg_room = err, room
        if mode != self.dbg_mode:
            self.debug("mode", f"DRIVE {self.dbg_mode} -> {mode}: sliding {err:+.0f} deg (robot frame, 0 = forwards, +90 = left), "
                       f"{room:.2f} m to the nearest wall on the map" + (", narrow" if self.narrow else ""), 0.2)
            self.dbg_mode = mode

    def path_bearing(self, ahead:float=0.4) -> float:
        '''Where the path heads: bearing (deg from the robot's heading, +ve left) of its first point more than `ahead` m away.'''
        pts, c = self.path_xy, self.centre
        far = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) > ahead
        p = pts[int(np.argmax(far))] if np.any(far) else pts[-1]
        return wrap_deg(math.degrees(math.atan2(p[1] - c[1], p[0] - c[0])) - self.cpose[2])

    def fmt_nearest(self) -> str:
        '''Nearest LiDAR return outside the footprint, in the robot frame (DEBUG prints).'''
        p = self.pts_r
        p = p[(p[:, 0] < -ROBOT_REAR) | (p[:, 0] > ROBOT_FRONT) | (np.abs(p[:, 1]) > ROBOT_HALF_WIDTH)]
        if len(p) == 0:
            return "none"
        k = int(np.argmin(np.hypot(p[:, 0] - CENTRE_X, p[:, 1])))
        return f"({p[k, 0]:+.2f}, {p[k, 1]:+.2f}) m robot frame = {math.degrees(math.atan2(p[k, 1], p[k, 0] - CENTRE_X)):+.0f} deg"

    def print_map(self, why:str="", half:float=1.5):
        '''DEBUG: ASCII map around the robot, 10 cm per character, course +x UP and +y LEFT (what the robot sees at the start).
        # wall seen by the LiDAR   + too close to a wall for the robot's centre   . seen free   (blank) never seen
        * A* path still to drive   ^ < v > the robot and where it faces   G goal   ~ outside the map bounds'''
        if not DEBUG or self.grid is None or self.centre is None:
            return
        g, k = self.grid, 2 # k cells per character
        n = int(round(half / (k * g.res)))
        H, W = g.shape
        ri, rj = (int(v) for v in g.cell(*self.centre))
        path = set() if self.path is None else {((i - ri) // k, (j - rj) // k) for i, j in self.path[self.path_idx:]}
        gi, gj = (int(v) for v in g.cell(*(self.path_goal or self.cfg["goal"])))
        goal = ((gi - ri) // k, (gj - rj) // k)
        robot = "^<v>"[int((self.cpose[2] + 45.0) % 360.0 // 90.0)]
        rows = []
        for a in range(n, -n - 1, -1): # +x up
            row = ""
            for b in range(n, -n - 1, -1): # +y left
                i0, j0 = ri + k * a, rj + k * b
                if a == 0 and b == 0:
                    row += robot
                elif (a, b) == goal:
                    row += "G"
                elif i0 < 0 or j0 < 0 or i0 >= H or j0 >= W:
                    row += "~"
                else:
                    blk = (slice(i0, i0 + k), slice(j0, j0 + k))
                    row += "#" if g.occupied[blk].any() else "*" if (a, b) in path else "+" if g.blocked[blk].any() else "." if (g.score[blk] < 0).any() else " "
            rows.append("|" + row + "|")
        self.last_map = self.clock()
        self.get_logger().info(f"DBG map{' (' + why + ')' if why else ''} at {self.fmt_pose()}, 10 cm/char, course +x up, +y left:\n" + "\n".join(rows))

    def log_status(self, now:float):
        '''Once a second, so it doesn't bury the state changes (like CA2's pose print).'''
        if now - self.last_log < 1.0:
            return
        self.last_log = now
        msg = f"[{self.state}] {self.fmt_pose()}"
        if self.state in ("NAVIGATE", "DOOR_ALIGN") and self.path is not None:
            msg += f" | path {len(self.path)} cells to {'goal' if self.path_reached_goal else 'closest reachable'}"
        if self.door is not None:
            msg += f" | door x={self.door['x']:.2f} y={self.door['y']:.2f} w={self.door['width']:.2f}"
        if self.tracker is not None and self.state.startswith("SWING"):
            line = self.tracker.lines[self.swing_k]
            if line["seen"]:
                msg += f" | plate {self.swing_k + 1} y={line['hist'][-1][1]:.2f} v={line['v']:+.2f} swing {line['ymin']:.2f}..{line['ymax']:.2f}"
            else:
                msg += f" | plate {self.swing_k + 1} not seen"
        self.get_logger().info(msg)
        if DEBUG and self.pose is not None and self.cpose is not None:
            m, cp = self.pose, self.cpose
            dbg = f"DBG POSE thinks it is at course ({cp[0]:+.3f}, {cp[1]:+.3f}, {cp[2]:+.1f} deg) = maze ({m[0]:+.3f}, {m[1]:+.3f}, {m[2]:+.1f} deg)"
            if self.dbg_pose_ref is not None:
                dt = now - self.dbg_pose_t
                moved = math.hypot(m[0] - self.dbg_pose_ref[0], m[1] - self.dbg_pose_ref[1])
                dbg += f" | last {dt:.1f} s: commands {self.dbg_cmd_dist:.2f} m, pose moved {moved:.2f} m"
                if not self.is_simulation:
                    dbg += f" | {self._pose_msgs / max(dt, 1e-3):.0f} Optitrack msgs/s"
                    if self.pose_time is not None:
                        dbg += f", value last changed {now - self.pose_time:.1f} s ago"
                if self.dbg_cmd_dist > 0.1 and moved < 0.3 * self.dbg_cmd_dist:
                    dbg += "  <-- POSE NOT FOLLOWING THE COMMANDS: Optitrack frozen/lost, or the wheels aren't moving the robot"
            self.get_logger().info(dbg)
            self.dbg_pose_ref, self.dbg_pose_t, self.dbg_cmd_dist, self._pose_msgs = m.copy(), now, 0.0, 0
        if DEBUG and self.pts_r is not None:
            c, s = self.dbg_cmd, self.dbg_sent
            dbg = f"DBG cmd vx {c[0]:+.2f} vy {c[1]:+.2f} turn {c[2]:+.2f}"
            if any(abs(a - b) > 1e-3 for a, b in zip(c, s)):
                dbg += f" -> safety sent vx {s[0]:+.2f} vy {s[1]:+.2f} turn {s[2]:+.2f}"
            if self.dbg_mode is not None:
                dbg += f" | sliding {self.dbg_mode} ({self.dbg_err:+.0f} deg), room {self.dbg_room:.2f} m"
            dbg += f" | nearest {self.fmt_nearest()} | {self.self_hits} self-hits dropped"
            if self.grid is not None:
                dbg += f" | map {int(self.grid.occupied.sum())} wall cells"
            self.get_logger().info(dbg)
            if DEBUG_MAP_PERIOD > 0 and self.mapping and now - self.last_map >= DEBUG_MAP_PERIOD:
                self.print_map()


def main(args=None):
    arg_parser = argparse.ArgumentParser(description="RB2301 final project")
    arg_parser.add_argument('--calibrate', action='store_true', help="Real robot: only print the Optitrack pose and the maze-frame pose twice a second, never drive")
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    print("Starting obstacle course")
    rclpy.init(args=ros_args)

    # Use lidar scan length to determine if in simulation or not
    lidar = LidarNode()
    while lidar.is_simulation is None:
        rclpy.spin_once(lidar)
    is_simulation = lidar.is_simulation
    lidar.destroy_node()

    # Start spinning the obstacle node and only stop once SystemExit error is raised within the node callback
    obstacle = ObstacleCourseNode(is_simulation, cli_args.calibrate)
    try:
        rclpy.spin(obstacle)
    except SystemExit:
        print("Shutting down")
    obstacle.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()