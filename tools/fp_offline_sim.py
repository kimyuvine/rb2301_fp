#!/usr/bin/env python3
'''Offline 2D test bench for rb2301_fp/obstacle_course.py -- no ROS or Gazebo needed (just numpy, matplotlib for --plot).

Rebuilds the final-project course from obstacle_course_world_fp_original.sdf and the randomisation in
obstacle_modifier.py, fakes the LiDAR (ray casting) and the swinging plates, integrates the velocities the node
publishes, and runs the real ObstacleCourseNode on it. It checks the logic, it is not a physics simulation:
no wheel slip, no acceleration limits, perfect odometry, and touching a wall ends the run instead of pushing back.

    python3 tools/fp_offline_sim.py                   # 20 randomised episodes
    python3 tools/fp_offline_sim.py -n 1 --seed 3 -v  # one episode, with the node's log
    python3 tools/fp_offline_sim.py --plot out/       # also save a picture of every failed episode (--plot-all: every one)
    python3 tools/fp_offline_sim.py --original        # the un-randomised map
    python3 tools/fp_offline_sim.py --real            # real-robot settings: Optitrack pose in the CA2 maze frame, 720-ray LiDAR, 0.3 m/s
'''
import argparse
import math
import os
import sys
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DT = 0.05 # controller period, as create_timer(0.05, ...) in the node

# --- Course (from obstacle_course_world_fp_original.sdf) -----------------------
# Wall model poses (x, y, yaw in degrees). Each "nist maze wall 120" is a 1.2 m plane 0.6 m to the model's +y side.
WALLS = [(0, 0, 0), (0, -1.2, 0), (0, 0, 90), (1.2, 0, 0), (1.2, -1.2, 0), (2.4, 0, 0), (2.4, -1.2, 90), (3.6, 0, 90),
         (2.4, -2.4, 90), (3.6, -1.2, 90), (2.4, -2.8, 90), (3.6, -2.4, 0), (2.4, -4.0, 0), (3.6, -4.0, 0), (4.8, -2.4, 0),
         (4.8, -4.0, 0), (6, -2.4, 90), (6, -2.8, 90)]
GATE_Y = -1.4567267699137467 # y of the left gate wall's pose in the original map; the right one is 1.6 m further -y
OBSTACLE_NAMES = ("coke", "drawer", "bin")
OBSTACLE_SIZE = {0: 0.03, 1: (0.205, 0.24), 2: 0.226} # coke / litter bin radius, drawer half sizes (from the meshes)

def wall_segment(x, y, yaw):
    if yaw == 0:
        return (x - 0.6, y + 0.6, x + 0.6, y + 0.6)
    return (x - 0.6, y - 0.6, x - 0.6, y + 0.6)

def box_segments(cx, cy, hx, hy):
    a, b, c, d = (cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)
    return [(*a, *b), (*b, *c), (*c, *d), (*d, *a)]

class Course:
    '''One randomised course, exactly as obstacle_modifier.py builds it (all randomise_* switches on unless original).'''
    def __init__(self, rng, original=False):
        u = rng.random
        self.xo = 0.0 if original else -u() * 0.8
        self.yo = 0.0 if original else -u() * 0.6
        gate_y = GATE_Y if original else -u() * 0.5 - 1.4
        segs = []
        for x, y, yaw in WALLS:
            if x > 1.0:
                x += self.xo
            if y < -2.6:
                y += self.yo
            segs.append(wall_segment(x, y, yaw))
        gy = gate_y + (0.0 if original else self.yo)
        gx = 4.2 + (0.0 if original else self.xo)
        segs += [wall_segment(gx, gy, 90), wall_segment(gx, gy - 1.6, 90)]
        self.door = (gx - 0.6, gy - 0.8) # line x, gap centre y (gap is 0.4 m)

        self.obstacles = [] # (type, x, y)
        base = [(0.6, 0.0, 0), (2.4 + self.xo, -0.6 + self.yo, 0), (2.4 + self.xo, -1.4 + self.yo, 1), (2.4 + self.xo, -2.4 + self.yo, 2)]
        circles = []
        for x, y, kind in base:
            if not original:
                x, y, kind = x + (u() - 0.5) * 0.8, y + (u() - 0.5) * 0.6, int(rng.integers(3))
            self.obstacles.append((kind, x, y))
            if kind == 1:
                segs += box_segments(x, y, *OBSTACLE_SIZE[1])
            else:
                circles.append((x, y, OBSTACLE_SIZE[kind]))
        self.static_segs = np.array(segs, dtype=float)
        self.circles = np.array(circles, dtype=float).reshape(-1, 3)

        self.plate_x = [4.2 + self.xo, 4.8 + self.xo]
        self.plate_y = (-2.0 + self.yo, -3.2 + self.yo)
        self.half_period = [4.0, 2.5] if original else [u() * 4.0 + 2.0, u() * 4.0 + 2.0]
        self.phase = [u() * 2 * T for T in self.half_period]
        self.end_x = 5.4 + self.xo

    def plate_centres(self, t):
        out = []
        for x, T, ph in zip(self.plate_x, self.half_period, self.phase):
            s = ((t + ph) % (2 * T)) / T
            s = s if s <= 1.0 else 2.0 - s
            out.append((x, self.plate_y[0] + (self.plate_y[1] - self.plate_y[0]) * s))
        return out

    def plate_segs(self, t):
        segs = []
        for x, y in self.plate_centres(t):
            segs += box_segments(x, y, 0.025, 0.15)
        return np.array(segs, dtype=float)

    def describe(self):
        obs = ", ".join(f"{OBSTACLE_NAMES[k]}({x:.2f},{y:.2f})" for k, x, y in self.obstacles)
        return (f"xo={self.xo:+.2f} yo={self.yo:+.2f} door=({self.door[0]:.2f},{self.door[1]:.2f}) "
                f"T=({self.half_period[0]:.1f},{self.half_period[1]:.1f})s  {obs}")

# --- Geometry -------------------------------------------------------------------
def cast(origin, angles, segs, circles, rmax=10.0):
    '''Distance along each ray from origin to the nearest segment (M, 4) or circle (K, 3); inf if beyond rmax.'''
    d = np.column_stack((np.cos(angles), np.sin(angles)))
    r = np.full(len(angles), np.inf)
    if len(segs):
        P, E = segs[:, :2], segs[:, 2:] - segs[:, :2]
        w = P - origin
        denom = d[:, 0, None] * E[None, :, 1] - d[:, 1, None] * E[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[None, :, 0] * E[None, :, 1] - w[None, :, 1] * E[None, :, 0]) / denom
            s = (w[None, :, 0] * d[:, 1, None] - w[None, :, 1] * d[:, 0, None]) / denom
        ok = (np.abs(denom) > 1e-12) & (t > 1e-6) & (s >= 0.0) & (s <= 1.0)
        r = np.minimum(r, np.where(ok, t, np.inf).min(axis=1))
    if len(circles):
        oc = origin - circles[:, :2]
        b = d @ oc.T
        disc = b ** 2 - ((oc ** 2).sum(axis=1) - circles[:, 2] ** 2)[None, :]
        with np.errstate(invalid="ignore"):
            t = -b - np.sqrt(disc)
        r = np.minimum(r, np.where((disc >= 0) & (t > 1e-6), t, np.inf).min(axis=1))
    r[r > rmax] = np.inf
    return r

def footprint(x, y, h, rear, front, hw, n=12):
    '''Points along the outline of the robot's footprint (world frame).'''
    xs, ys = np.linspace(-rear, front, n), np.linspace(-hw, hw, n)
    local = np.vstack([np.column_stack((xs, np.full(n, -hw))), np.column_stack((xs, np.full(n, hw))),
                       np.column_stack((np.full(n, -rear), ys)), np.column_stack((np.full(n, front), ys))])
    c, s = math.cos(h), math.sin(h)
    return np.column_stack((x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]))

def clearance(points, segs, circles):
    '''Smallest distance from any of the points to a segment or circle edge (negative: a point is inside a circle).'''
    best = np.inf
    if len(segs):
        P, E = segs[:, :2], segs[:, 2:] - segs[:, :2]
        L2 = np.maximum((E ** 2).sum(axis=1), 1e-12)
        s = np.clip(((points[:, None, :] - P[None]) * E[None]).sum(axis=2) / L2[None], 0.0, 1.0)
        q = P[None] + s[..., None] * E[None]
        best = min(best, float(np.sqrt(((points[:, None, :] - q) ** 2).sum(axis=2)).min()))
    if len(circles):
        dc = np.sqrt(((points[:, None, :] - circles[None, :, :2]) ** 2).sum(axis=2)) - circles[None, :, 2]
        best = min(best, float(dc.min()))
    return best

# --- ROS stand-ins ----------------------------------------------------------------
class SimLog:
    def __init__(self, verbose):
        self.verbose, self.t, self.lines = verbose, 0.0, []
    def __call__(self, level, msg):
        line = f"[{self.t:6.2f}] {level:5s} {msg}"
        self.lines.append(line)
        if self.verbose:
            print(line)

def install_ros_stubs(log):
    class Vec:
        def __init__(self):
            self.x = self.y = self.z = 0.0
    class Quat(Vec):
        def __init__(self):
            super().__init__()
            self.w = 1.0
    class Pose:
        def __init__(self):
            self.position, self.orientation = Vec(), Quat()
    class Twist:
        def __init__(self):
            self.linear, self.angular = Vec(), Vec()
    class PoseStamped:
        def __init__(self):
            self.pose = Pose()
    class Odometry:
        def __init__(self):
            self.pose = types.SimpleNamespace(pose=Pose())
    class LaserScan:
        def __init__(self):
            self.ranges = []
    class Logger:
        def info(self, msg): log("INFO", msg)
        def warn(self, msg): log("WARN", msg)
        warning = warn
        def error(self, msg): log("ERROR", msg)
        def debug(self, msg): pass
    class Publisher:
        def __init__(self):
            self.msg = None
        def publish(self, msg):
            self.msg = msg
    class Node:
        def __init__(self, name):
            self._logger = Logger()
        def get_logger(self): return self._logger
        def create_subscription(self, *a, **k): return None
        def create_publisher(self, *a, **k): return Publisher()
        def create_timer(self, *a, **k): return None
        def destroy_node(self): pass

    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m
    module("rclpy", init=lambda args=None: None, spin=lambda n: None, spin_once=lambda n: None, shutdown=lambda: None)
    module("rclpy.node", Node=Node)
    module("rclpy.logging", set_logger_level=lambda *a, **k: None, LoggingSeverity=types.SimpleNamespace(INFO=20, DEBUG=10))
    module("rclpy.qos", ReliabilityPolicy=types.SimpleNamespace(BEST_EFFORT=2), QoSProfile=lambda **k: None)
    for pkg, names in (("geometry_msgs", {"PoseStamped": PoseStamped, "Twist": Twist}), ("nav_msgs", {"Odometry": Odometry}),
                       ("sensor_msgs", {"LaserScan": LaserScan})):
        module(pkg)
        module(pkg + ".msg", **names)
    return types.SimpleNamespace(PoseStamped=PoseStamped, Odometry=Odometry, LaserScan=LaserScan)

def quaternion(msg_orientation, yaw):
    msg_orientation.w, msg_orientation.z = math.cos(yaw / 2), math.sin(yaw / 2)

# --- Episode ----------------------------------------------------------------------
def run_episode(oc, msgs, log, seed, args):
    rng = np.random.default_rng(seed)
    course = Course(rng, original=args.original)
    log.t, log.lines = 0.0, []
    node = oc.ObstacleCourseNode(is_simulation=not args.real)
    t = 0.0
    node.clock = lambda: t

    # Robot pose in the course frame (= Gazebo world frame); x, y of the rear axle, heading in radians
    x, y, h = 0.0, 0.0, 0.0
    if args.start is not None:
        x, y = args.start
    if args.real: # placed by hand: a little off the corridor axis
        h = math.radians(rng.uniform(-4.0, 4.0))
        place = (rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5), rng.choice([0.0, 90.0]) + rng.uniform(-3.0, 3.0)) # course in the CA2 maze frame
        frame, _ = oc.load_frame(node.robot_number)
        n_rays, start_deg, step_deg = 720, 0.0, 0.5
    else:
        n_rays, start_deg, step_deg = 360, oc.SIM_CONFIG["lidar_start_deg"], oc.SIM_CONFIG["lidar_step_deg"]
    ray_angles = np.deg2rad(start_deg + step_deg * np.arange(n_rays))

    trail, states = [], []
    result, min_clear, plate_hits, in_plate = "TIMEOUT", np.inf, 0, False
    tick = 0
    if clearance(footprint(x, y, h, oc.ROBOT_REAR, oc.ROBOT_FRONT, oc.ROBOT_HALF_WIDTH), course.static_segs, course.circles) < 0.01:
        result, t = "INVALID", args.timeout # the randomiser put the start-corridor obstacle on top of the robot
    while t < args.timeout:
        log.t = t
        # Sensors: pose every tick, LiDAR at 10 Hz
        if args.real:
            mx, my, mh = place_pose(x, y, math.degrees(h), place)
            r = math.radians(frame["rotation_deg"])
            msg = msgs.PoseStamped()
            msg.pose.position.x = frame["origin_x"] + math.cos(r) * mx - math.sin(r) * my + rng.normal(0, 0.002)
            msg.pose.position.y = frame["origin_y"] + math.sin(r) * mx + math.cos(r) * my + rng.normal(0, 0.002)
            quaternion(msg.pose.orientation, math.radians(mh + frame["rotation_deg"] + frame["heading_offset_deg"] + rng.normal(0, 0.3)))
            node.optitrack_callback(msg)
        else:
            msg = msgs.Odometry()
            msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
            quaternion(msg.pose.pose.orientation, h)
            node.odometer_callback(msg)
        if tick % 2 == 0:
            lx, ly = x + oc.LIDAR_X * math.cos(h), y + oc.LIDAR_X * math.sin(h)
            segs = np.vstack([course.static_segs, course.plate_segs(t)])
            ranges = cast(np.array((lx, ly)), h + ray_angles, segs, course.circles)
            if args.real:
                ranges = ranges + rng.normal(0, 0.01, len(ranges))
                ranges[rng.random(len(ranges)) < 0.01] = 0.0 # dropouts
            scan = msgs.LaserScan()
            scan.ranges = ranges.tolist()
            node.sub_scan_callback(scan)

        node.timer_callback()
        cmd = node.publisher_.msg
        vx, vy, w = (cmd.linear.x, cmd.linear.y, cmd.angular.z) if cmd is not None else (0.0, 0.0, 0.0)

        # Integrate the command (body twist about the rear axle, as Gazebo's VelocityControl does)
        x += (vx * math.cos(h) - vy * math.sin(h)) * DT
        y += (vx * math.sin(h) + vy * math.cos(h)) * DT
        h += w * DT
        t += DT
        tick += 1
        trail.append((x, y, h))
        states.append(node.state)

        outline = footprint(x, y, h, oc.ROBOT_REAR, oc.ROBOT_FRONT, oc.ROBOT_HALF_WIDTH)
        c = clearance(outline, course.static_segs, course.circles)
        min_clear = min(min_clear, c)
        if c < 0.003:
            result = "COLLISION"
            break
        touching = clearance(outline, course.plate_segs(t), np.empty((0, 3))) < 0.003
        plate_hits += touching and not in_plate
        in_plate = touching
        if node.state == "DONE":
            rear_x = outline[:, 0].min()
            result = "OK" if rear_x > course.plate_x[1] + 0.025 else "SHORT"
            break
    return {"seed": seed, "course": course, "node": node, "result": result, "time": t, "min_clear": min_clear,
            "plate_hits": plate_hits, "trail": np.array(trail).reshape(-1, 3), "states": states, "log": list(log.lines)}

def place_pose(x, y, h_deg, place):
    '''Course-frame pose -> CA2 maze frame, for a course whose origin sits at place = (x, y, rotation deg) in the maze.'''
    r = math.radians(place[2])
    return (place[0] + math.cos(r) * x - math.sin(r) * y, place[1] + math.sin(r) * x + math.cos(r) * y, h_deg + place[2])

def plot_episode(ep, path, oc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    course, node, trail = ep["course"], ep["node"], ep["trail"]
    fig, ax = plt.subplots(figsize=(7, 8))
    if node.grid is not None:
        occ = np.argwhere(node.grid.occupied)
        if len(occ):
            gx, gy = node.grid.centre(occ[:, 0], occ[:, 1])
            ax.scatter(gx, gy, s=4, c="#bbbbbb", marker="s", label="node's map")
    for s in course.static_segs:
        ax.plot([s[0], s[2]], [s[1], s[3]], "k-", lw=1.5)
    for cx, cy, r in course.circles:
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, color="k"))
    for px in course.plate_x:
        ax.plot([px, px], [course.plate_y[0] - 0.15, course.plate_y[1] + 0.15], "r:", lw=1)
    if node.path_xy is not None:
        ax.plot(node.path_xy[:, 0], node.path_xy[:, 1], "g-", lw=1, label="last A* path")
    ax.plot(trail[:, 0], trail[:, 1], "b-", lw=1, label="rear axle")
    out = footprint(*trail[-1], oc.ROBOT_REAR, oc.ROBOT_FRONT, oc.ROBOT_HALF_WIDTH)
    ax.plot(out[:, 0], out[:, 1], "b.", ms=1)
    if node.door is not None:
        ax.plot(node.door["x"], node.door["y"], "m*", ms=10, label="door estimate")
    ax.set_aspect("equal")
    ax.set_title(f"seed {ep['seed']}: {ep['result']} ({node.state}) t={ep['time']:.1f}s")
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="seed of the first episode (episode i uses seed + i)")
    ap.add_argument("--original", action="store_true", help="un-randomised map")
    ap.add_argument("--real", action="store_true", help="real-robot settings (Optitrack, 720-ray noisy LiDAR, 0.3 m/s)")
    ap.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), help="start position, like x:= y:= in gz_fp.sh")
    ap.add_argument("--timeout", type=float, default=150.0, help="seconds of simulated time per episode")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the node's log")
    ap.add_argument("--plot", metavar="DIR", help="save a picture of each failed episode here")
    ap.add_argument("--plot-all", action="store_true", help="with --plot: every episode, not just failures")
    args = ap.parse_args()

    log = SimLog(args.verbose)
    msgs = install_ros_stubs(log)
    sys.path.insert(0, REPO)
    from rb2301_fp import obstacle_course as oc

    if args.plot:
        os.makedirs(args.plot, exist_ok=True)
    counts = {}
    for i in range(args.episodes):
        ep = run_episode(oc, msgs, log, args.seed + i, args)
        counts[ep["result"]] = counts.get(ep["result"], 0) + 1
        print(f"seed {ep['seed']:3d}  {ep['result']:9s} t={ep['time']:5.1f}s  clear={ep['min_clear']:.3f}m  plate_hits={ep['plate_hits']}  "
              f"recoveries={ep['node'].recoveries}  end={ep['node'].state:11s}  {ep['course'].describe()}")
        if ep["result"] not in ("OK", "INVALID") and not args.verbose:
            for line in ep["log"][-6:]:
                print("      " + line)
        if args.plot and len(ep["trail"]) and (args.plot_all or ep["result"] not in ("OK", "INVALID")):
            plot_episode(ep, os.path.join(args.plot, f"seed{ep['seed']:03d}_{ep['result']}.png"), oc)
    print("summary:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), f"of {args.episodes}")

if __name__ == "__main__":
    main()
