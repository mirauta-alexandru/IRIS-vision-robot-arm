#!/usr/bin/env python3
"""
IRIS Vision v3 — solvePnP + Ray-Plane Intersection

Instead of homography (2D→2D, extrapolates badly), this uses:
  1. Camera calibration (intrinsics) — you already have this
  2. solvePnP → exact 3D pose of ChArUco board relative to camera
  3. Known board offset → camera-to-robot transform
  4. Ray-plane intersection → click anywhere, get robot coords

Accurate across the ENTIRE frame, not just where the board is.

Keys:
  SPACE = capture board & calibrate camera→robot transform
  F = flip axis mapping (rotate 90°)
  T = test mode (show coords without moving)
  G = toggle grid overlay
  E = toggle error overlay
  S = save calibration
  +/- = adjust pick Z
  V = verify mode (move board around, see error in real-time)
  R = recalibrate
  Q/ESC = quit
"""

import cv2
import numpy as np
import json
import os
import sys
import requests
import threading
import queue
import time
import math

# ─── CONFIG ───
CAMERA_ID = 0
BRIDGE_URL = "http://localhost:8765"
PICK_Z = 100
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
calib_path = os.path.join(SAVE_DIR, "camera_calibration.json")
npz_path = os.path.join(SAVE_DIR, "camera_calibration.npz")
transform_path = os.path.join(SAVE_DIR, "vision_calibration_v3.json")
correction_path = os.path.join(SAVE_DIR, "vision_correction.json")
correction_debug_path = os.path.join(SAVE_DIR, "vision_correction_debug.png")
BOARD_ONLY_MODE = True
MIN_CHARUCO_CORNERS = 12

# Board config
BOARD_SQUARES = (7, 5)
SQUARE_SIZE = 35.0        # mm
MARKER_SIZE = 25.0        # mm

# Workspace bounds (mm, robot frame)
WORKSPACE_X_MIN = 80
WORKSPACE_X_MAX = 350
WORKSPACE_Y_MIN = -120
WORKSPACE_Y_MAX = 120

# Gripper marker correction config
GRIPPER_MARKER_ID = 21
GRIPPER_MARKER_DICT = cv2.aruco.DICT_4X4_50
GRIPPER_MARKER_SIZE_MM = 30.0
MARKER_BEHIND_TIP_MM = 120.0
CORRECTION_Z_MM = 45.0
CORRECTION_WRIST_ROLL_DEG = 55
CORRECTION_WRIST_PITCH_DEG = 27
CORRECTION_WRIST_PITCH_CANDIDATES = [27, 40, 55, 70, 85, 15, 100, 120]
CORRECTION_SETTLE_SECONDS = 2.6
CORRECTION_DETECTIONS_PER_POINT = 5
MAX_REASONABLE_CORRECTION_ERROR_MM = 35.0
CORRECTION_TARGETS = [
    (150, -70), (150, 0), (150, 70),
    (210, -90), (210, 0), (210, 90),
    (270, -90), (270, 0), (270, 90),
    (325, -65), (325, 0), (325, 65),
]

# ─── State ───
mode = "detect"  # detect → pick
last_click = None
show_grid = True
show_errors = False
verify_mode = False
test_mode = False
axis_mapping = 0
board_offset_x = 85.0
board_offset_y = 0.0
avg_error = None
input_q = queue.Queue()
latest_frame = None
latest_frame_lock = threading.Lock()
correction_running = False
correction_status = ""
waiting_for_manual_correction = False
manual_correction_anchor = None

# 3D calibration state
cam_matrix = None
dist_coeffs = None
new_cam_matrix = None
use_undistort = False
remap_map1 = None
remap_map2 = None

# Camera-to-robot transform (computed from single board capture)
# R_cam2robot and t_cam2robot transform a point from camera frame to robot frame
R_cam2robot = None
t_cam2robot = None
# The table plane in camera coords: normal vector and point on plane
table_normal_cam = None
table_point_cam = None
vision_correction = None


# ─── Axis mapping configs ───
AXIS_CONFIGS = [
    {"label": "Board cols → Robot +X, rows → Robot +Y",  "x_sign": 1, "y_sign": 1,  "swap": False},
    {"label": "Board cols → Robot +Y, rows → Robot -X",  "x_sign": -1, "y_sign": 1, "swap": True},
    {"label": "Board cols → Robot -X, rows → Robot -Y",  "x_sign": -1, "y_sign": -1, "swap": False},
    {"label": "Board cols → Robot -Y, rows → Robot +X",  "x_sign": 1, "y_sign": -1,  "swap": True},
]


def board_corner_to_robot(col, row, cfg):
    """Convert a board grid position to robot coordinates."""
    local_x = col * SQUARE_SIZE
    local_y = row * SQUARE_SIZE
    if cfg["swap"]:
        rx = cfg["x_sign"] * local_y + board_offset_x
        ry = cfg["y_sign"] * local_x + board_offset_y
    else:
        rx = cfg["x_sign"] * local_x + board_offset_x
        ry = cfg["y_sign"] * local_y + board_offset_y
    return rx, ry


def compute_camera_to_robot(charuco_corners, charuco_ids):
    """
    From a single board capture, compute the full camera→robot transform.
    
    Steps:
    1. solvePnP → board pose in camera frame (R_board2cam, t_board2cam)
    2. Board corners → robot coords (known from offset + axis mapping)
    3. Solve rigid transform: camera frame → robot frame
    4. Extract table plane in camera coords for ray-plane intersection
    """
    global R_cam2robot, t_cam2robot, table_normal_cam, table_point_cam, avg_error

    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < MIN_CHARUCO_CORNERS:
        count = 0 if charuco_corners is None else len(charuco_corners)
        print(f"  ✗ Need at least {MIN_CHARUCO_CORNERS} ChArUco corners for board-only calibration (have {count})")
        return False

    num_cols = BOARD_SQUARES[0] - 1
    cfg = AXIS_CONFIGS[axis_mapping]

    # 3D object points of detected corners (board-local, in meters for solvePnP)
    obj_pts_3d = []
    robot_pts = []
    for cid in charuco_ids.flatten():
        col = cid % num_cols
        row = cid // num_cols
        # Board-local 3D (Z=0 on board plane), in meters
        obj_pts_3d.append([col * SQUARE_SIZE / 1000.0, row * SQUARE_SIZE / 1000.0, 0.0])
        # Robot coords (mm)
        rx, ry = board_corner_to_robot(col, row, cfg)
        robot_pts.append([rx, ry, 0.0])  # Z=0 = table plane

    obj_pts_3d = np.array(obj_pts_3d, dtype=np.float64)
    robot_pts = np.array(robot_pts, dtype=np.float64)
    img_pts = charuco_corners.reshape(-1, 2).astype(np.float64)

    # Use undistorted camera matrix if available, with zero distortion
    if use_undistort:
        solve_cam = new_cam_matrix if new_cam_matrix is not None else cam_matrix
        solve_dist = np.zeros(5)
    else:
        solve_cam = cam_matrix
        solve_dist = dist_coeffs

    # solvePnP: get board pose in camera frame
    success, rvec, tvec = cv2.solvePnP(obj_pts_3d, img_pts, solve_cam, solve_dist)
    if not success:
        print("  ✗ solvePnP failed!")
        return False

    # Convert to rotation matrix
    R_board2cam, _ = cv2.Rodrigues(rvec)
    t_board2cam = tvec.flatten()  # translation in meters

    # Now we have: point_cam = R_board2cam @ point_board + t_board2cam
    # We need: point_robot = R_cam2robot @ point_cam + t_cam2robot
    
    # Compute camera-frame coordinates of each board corner
    cam_pts = []
    for i, obj_pt in enumerate(obj_pts_3d):
        pt_cam = R_board2cam @ obj_pt + t_board2cam
        cam_pts.append(pt_cam)
    cam_pts = np.array(cam_pts)

    # Robot points in meters for consistent units
    robot_pts_m = robot_pts / 1000.0

    # Solve rigid transform: robot = R @ cam + t
    # Using least squares with SVD (Procrustes)
    cam_centroid = cam_pts.mean(axis=0)
    robot_centroid = robot_pts_m.mean(axis=0)

    cam_centered = cam_pts - cam_centroid
    robot_centered = robot_pts_m - robot_centroid

    H = cam_centered.T @ robot_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = robot_centroid - R @ cam_centroid

    R_cam2robot = R
    t_cam2robot = t

    # Table plane in camera frame
    # The board lies on the table, so any 3 board points define the table plane
    # Normal = cross product of two edge vectors
    v1 = cam_pts[1] - cam_pts[0]
    v2 = cam_pts[-1] - cam_pts[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    # Make sure normal points toward camera (positive Z in camera frame)
    if normal[2] < 0:
        normal = -normal

    table_normal_cam = normal
    table_point_cam = cam_pts[0]

    # ─── Verify: compute reprojection errors ───
    errors = []
    for i in range(len(cam_pts)):
        # Camera → robot
        robot_pred = (R_cam2robot @ cam_pts[i] + t_cam2robot) * 1000.0  # back to mm
        robot_actual = robot_pts[i]  # already in mm
        err = np.linalg.norm(robot_pred - robot_actual)
        errors.append(err)

    avg_error = np.mean(errors)
    max_error = np.max(errors)

    print()
    print(f"  ── Camera→Robot Calibration (solvePnP) ──")
    print(f"  Axis: {cfg['label']}")
    print(f"  Offset: X={board_offset_x}mm Y={board_offset_y}mm")
    print(f"  Points: {len(cam_pts)}")
    print(f"  Avg error: {avg_error:.3f}mm {'✓' if avg_error < 1 else '~' if avg_error < 2 else '✗'}")
    print(f"  Max error: {max_error:.3f}mm")

    # Show a few sample points
    print(f"  ── Sample points ──")
    check_indices = [0, len(img_pts) // 2, -1]
    for idx in check_indices:
        cid = charuco_ids.flatten()[idx]
        rp = robot_pts[idx]
        pred = (R_cam2robot @ cam_pts[idx] + t_cam2robot) * 1000.0
        err = errors[idx]
        print(f"    Corner {cid}: robot({rp[0]:.0f},{rp[1]:.0f}) pred({pred[0]:.1f},{pred[1]:.1f}) err={err:.2f}mm")

    # Camera position in robot frame
    cam_pos_robot = (R_cam2robot @ np.zeros(3) + t_cam2robot) * 1000.0
    print(f"  Camera position (robot frame): X={cam_pos_robot[0]:.0f} Y={cam_pos_robot[1]:.0f} Z={cam_pos_robot[2]:.0f}mm")

    return True


def load_vision_correction():
    global vision_correction
    vision_correction = None
    if BOARD_ONLY_MODE:
        return
    if not os.path.exists(correction_path):
        return
    try:
        with open(correction_path) as f:
            data = json.load(f)
        points = data.get("points", [])
        if len(points) < 3:
            print("⚠ Vision correction ignored: need 3+ points")
            return
        vision_correction = {
            "points": points,
            "power": float(data.get("power", 2.0)),
            "max_radius_mm": float(data.get("max_radius_mm", 220.0)),
            "avg_error_before": data.get("avg_error_before"),
            "avg_error_after": data.get("avg_error_after"),
        }
        print(f"✓ Vision correction loaded ({len(points)} pts)")
    except Exception as e:
        vision_correction = None
        print(f"⚠ Vision correction error: {e}")


def apply_vision_correction(rx, ry):
    if BOARD_ONLY_MODE:
        return rx, ry
    if not vision_correction:
        return rx, ry

    samples = vision_correction["points"]
    power = vision_correction["power"]
    max_radius = vision_correction["max_radius_mm"]
    weighted_dx = 0.0
    weighted_dy = 0.0
    total_w = 0.0

    for sample in samples:
        ox, oy = sample["observed_tip_xy"]
        tx, ty = sample["true_tip_xy"]
        dx = float(rx) - float(ox)
        dy = float(ry) - float(oy)
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return float(tx), float(ty)
        if dist > max_radius:
            continue
        residual_x = float(tx) - float(ox)
        residual_y = float(ty) - float(oy)
        w = 1.0 / max(dist, 1.0) ** power
        weighted_dx += residual_x * w
        weighted_dy += residual_y * w
        total_w += w

    if total_w <= 0:
        return rx, ry
    return float(rx) + weighted_dx / total_w, float(ry) + weighted_dy / total_w


def pixel_to_robot(px, py, z_robot_mm=0.0, apply_correction=True):
    """
    Convert pixel coordinates to robot coordinates via ray-plane intersection.
    
    1. Pixel → normalized camera ray
    2. Ray-plane intersection with table plane (in camera frame)
    3. Camera frame → robot frame
    
    z_robot_mm: height above table plane (0 = on table)
    """
    if R_cam2robot is None or table_normal_cam is None:
        return None

    # Use correct camera matrix
    if use_undistort:
        K = new_cam_matrix if new_cam_matrix is not None else cam_matrix
    else:
        K = cam_matrix

    # Pixel to normalized ray in camera frame
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([px, py, 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    # Ray-plane intersection
    # Plane: dot(normal, (P - table_point)) = 0
    # Ray: P = t * ray_cam (origin at camera = [0,0,0])
    denom = np.dot(table_normal_cam, ray_cam)
    if abs(denom) < 1e-8:
        return None  # Ray parallel to plane

    t = np.dot(table_normal_cam, table_point_cam) / denom
    if t < 0:
        return None  # Intersection behind camera

    # 3D point on table in camera frame
    point_cam = t * ray_cam

    # Transform to robot frame (meters)
    point_robot_m = R_cam2robot @ point_cam + t_cam2robot

    # Convert to mm
    rx = point_robot_m[0] * 1000.0
    ry = point_robot_m[1] * 1000.0
    rz = point_robot_m[2] * 1000.0

    # Adjust for Z offset (if picking above table)
    # The intersection was with Z=0 (table). If z_robot_mm > 0,
    # we need to intersect with a higher plane.
    if z_robot_mm != 0.0:
        # Offset the table plane in camera frame
        offset_m = z_robot_mm / 1000.0
        # Move table_point along the normal (in robot frame, Z is up)
        # Robot Z axis in camera frame:
        z_robot_in_cam = np.linalg.inv(R_cam2robot)[:, 2]  # 3rd column of R_robot2cam
        adjusted_point_cam = table_point_cam - z_robot_in_cam * offset_m
        t_adj = np.dot(table_normal_cam, adjusted_point_cam) / denom
        if t_adj > 0:
            point_cam_adj = t_adj * ray_cam
            point_robot_adj = R_cam2robot @ point_cam_adj + t_cam2robot
            rx = point_robot_adj[0] * 1000.0
            ry = point_robot_adj[1] * 1000.0

    if apply_correction:
        return apply_vision_correction(rx, ry)
    return rx, ry


def radial_unit(x_mm, y_mm):
    norm = math.hypot(x_mm, y_mm)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    return np.array([x_mm / norm, y_mm / norm], dtype=np.float64)


def tip_from_marker(marker_xy, true_tip_xy):
    """Marker is behind the tip toward base, so tip = marker + radial * distance."""
    return np.array(marker_xy, dtype=np.float64) + radial_unit(true_tip_xy[0], true_tip_xy[1]) * MARKER_BEHIND_TIP_MM


def robot_to_pixel(rx_mm, ry_mm, rz_mm=0.0):
    """Convert robot coordinates back to pixel coordinates (for grid overlay)."""
    if R_cam2robot is None:
        return None

    if use_undistort:
        K = new_cam_matrix if new_cam_matrix is not None else cam_matrix
    else:
        K = cam_matrix

    # Robot to camera frame
    point_robot_m = np.array([rx_mm / 1000.0, ry_mm / 1000.0, rz_mm / 1000.0])
    R_robot2cam = R_cam2robot.T
    t_robot2cam = -R_cam2robot.T @ t_cam2robot
    point_cam = R_robot2cam @ point_robot_m + t_robot2cam

    if point_cam[2] <= 0:
        return None  # Behind camera

    # Project to pixel
    px = K @ point_cam
    return int(px[0] / px[2]), int(px[1] / px[2])


def is_in_workspace(rx, ry):
    margin = 150
    return (WORKSPACE_X_MIN - margin <= rx <= WORKSPACE_X_MAX + margin and
            WORKSPACE_Y_MIN - margin <= ry <= WORKSPACE_Y_MAX + margin)


def send_ik(x, y, z):
    try:
        r = requests.post(f"{BRIDGE_URL}/ik",
                          json={"x": float(x), "y": float(y), "z": float(z), "send": True}, timeout=3)
        d = r.json()
        return d.get("error_mm", -1) if d.get("ok") else -1
    except:
        return -1


def move_robot(x, y, z, wrist_roll_deg=None, wrist_pitch_deg=None, timeout=8):
    payload = {"x": float(x), "y": float(y), "z": float(z), "send": True}
    if wrist_roll_deg is not None:
        payload["wrist_roll_deg"] = float(wrist_roll_deg)
    if wrist_pitch_deg is not None:
        payload["wrist_pitch_deg"] = float(wrist_pitch_deg)
    r = requests.post(f"{BRIDGE_URL}/ik", json=payload, timeout=timeout)
    d = r.json()
    if not d.get("ok"):
        raise RuntimeError(d.get("error", "IK failed"))
    actual = d.get("actual", {})
    return np.array([actual.get("x", x), actual.get("y", y)], dtype=np.float64)


def set_servo(ch, angle):
    requests.get(f"{BRIDGE_URL}/servo?ch={int(ch)}&angle={int(round(angle))}", timeout=2)


def get_robot_pose_xy(timeout=2):
    r = requests.get(f"{BRIDGE_URL}/pose", timeout=timeout)
    d = r.json()
    if not d.get("ok"):
        raise RuntimeError(d.get("error", "Could not read robot pose"))
    actual = d.get("actual", {})
    return np.array([actual["x"], actual["y"]], dtype=np.float64)


def get_latest_frame():
    with latest_frame_lock:
        return latest_frame.copy() if latest_frame is not None else None


def detect_gripper_marker_once(frame):
    dictionary_g = cv2.aruco.getPredefinedDictionary(GRIPPER_MARKER_DICT)
    detector = cv2.aruco.ArucoDetector(dictionary_g, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, corners, ids, None
    best_pts = None
    best_area = -1.0
    for idx, marker_id in enumerate(ids.flatten()):
        if int(marker_id) == GRIPPER_MARKER_ID:
            pts = corners[idx].reshape(4, 2)
            area = float(cv2.contourArea(pts.astype(np.float32)))
            if area > best_area:
                best_area = area
                best_pts = pts
    if best_pts is not None:
        return best_pts.mean(axis=0), corners, ids, best_pts
    return None, corners, ids, None


def detect_gripper_marker_average():
    centers = []
    marker_pts_samples = []
    last_frame = None
    last_corners = None
    last_ids = None
    for _ in range(45):
        frame = get_latest_frame()
        if frame is None:
            time.sleep(0.08)
            continue
        center, corners, ids, marker_pts = detect_gripper_marker_once(frame)
        last_frame, last_corners, last_ids = frame, corners, ids
        if center is not None:
            centers.append(center)
            marker_pts_samples.append(marker_pts)
            if len(centers) >= CORRECTION_DETECTIONS_PER_POINT:
                return (np.mean(np.array(centers), axis=0),
                        np.mean(np.array(marker_pts_samples), axis=0),
                        frame, corners, ids)
        time.sleep(0.10)
    return None, None, last_frame, last_corners, last_ids


def estimate_marker_pose(marker_pts):
    """Return marker pose in camera coordinates."""
    if R_cam2robot is None:
        return None

    s = GRIPPER_MARKER_SIZE_MM / 1000.0
    obj_pts = np.array([
        [-s / 2,  s / 2, 0.0],
        [ s / 2,  s / 2, 0.0],
        [ s / 2, -s / 2, 0.0],
        [-s / 2, -s / 2, 0.0],
    ], dtype=np.float64)
    img_pts = np.array(marker_pts, dtype=np.float64).reshape(4, 2)
    solve_cam = new_cam_matrix if use_undistort and new_cam_matrix is not None else cam_matrix
    solve_dist = np.zeros(5) if use_undistort else dist_coeffs
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, solve_cam, solve_dist)
    if not ok:
        return None

    R_m2c, _ = cv2.Rodrigues(rvec)
    t_m2c = tvec.flatten()
    return R_m2c, t_m2c


def marker_pose_to_robot_tip_xy(marker_pose, offset_m):
    R_m2c, t_m2c = marker_pose
    tip_cam = R_m2c @ offset_m + t_m2c
    tip_robot = (R_cam2robot @ tip_cam + t_cam2robot) * 1000.0
    return tip_robot[:2]


def solve_marker_tip_offset(samples):
    """Learn the constant marker-local vector from marker center to gripper tip."""
    a_rows = []
    b_rows = []
    for sample in samples:
        R_m2c = np.array(sample["marker_R_m2c"], dtype=np.float64)
        t_m2c = np.array(sample["marker_t_m2c"], dtype=np.float64)
        true_tip_xy_m = np.array(sample["true_tip_xy"], dtype=np.float64) / 1000.0
        marker_center_robot_m = R_cam2robot @ t_m2c + t_cam2robot
        transform_xy = (R_cam2robot @ R_m2c)[:2, :]
        a_rows.append(transform_xy)
        b_rows.append(true_tip_xy_m - marker_center_robot_m[:2])

    A = np.vstack(a_rows)
    b = np.concatenate(b_rows)
    offset_m, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)

    errors = []
    for sample in samples:
        marker_pose = (
            np.array(sample["marker_R_m2c"], dtype=np.float64),
            np.array(sample["marker_t_m2c"], dtype=np.float64),
        )
        observed_tip = marker_pose_to_robot_tip_xy(marker_pose, offset_m)
        true_tip = np.array(sample["true_tip_xy"], dtype=np.float64)
        errors.append(float(np.linalg.norm(true_tip - observed_tip)))

    return offset_m, rank, errors


def idw_correct(xy, points, exclude_index=None, power=2.0, max_radius_mm=220.0):
    weighted = np.zeros(2, dtype=np.float64)
    total = 0.0
    xy = np.array(xy, dtype=np.float64)
    for i, sample in enumerate(points):
        if exclude_index is not None and i == exclude_index:
            continue
        observed = np.array(sample["observed_tip_xy"], dtype=np.float64)
        true = np.array(sample["true_tip_xy"], dtype=np.float64)
        dist = float(np.linalg.norm(xy - observed))
        if dist < 1e-6:
            return true
        if dist > max_radius_mm:
            continue
        residual = true - observed
        weight = 1.0 / max(dist, 1.0) ** power
        weighted += residual * weight
        total += weight
    if total <= 0:
        return xy
    return xy + weighted / total


def run_gripper_correction_calibration():
    global vision_correction, correction_running, correction_status
    correction_running = True
    correction_status = "starting"
    if R_cam2robot is None:
        print("  ✗ Calibrate board first: show ChArUco, press SPACE, then S, then C")
        correction_running = False
        correction_status = "idle"
        return False

    print()
    print("═" * 62)
    print("  Gripper Marker Correction Calibration")
    print("═" * 62)
    print(f"  Marker: ArUco 4x4_50 ID {GRIPPER_MARKER_ID}")
    print(f"  Offset distance: marker center → gripper tip = {MARKER_BEHIND_TIP_MM:.0f}mm")
    print(f"  Z: {CORRECTION_Z_MM:.0f}mm | wrist roll: {CORRECTION_WRIST_ROLL_DEG}°")
    print(f"  Trying wrist pitch angles: {CORRECTION_WRIST_PITCH_CANDIDATES}")
    print("  Keep clear of the robot. Press Q in the camera window to abort only after this run finishes.")

    try:
        get_robot_pose_xy()
    except Exception as e:
        print()
        print("  ✗ Bridge does not expose current robot pose yet.")
        print(f"    {e}")
        print("    Restart iris_bridge.py, then run C again.")
        correction_running = False
        correction_status = "failed"
        return False

    samples = []
    debug_frame = None

    for idx, (tx, ty) in enumerate(CORRECTION_TARGETS, 1):
        print()
        print(f"  [{idx:02d}/{len(CORRECTION_TARGETS)}] Move tip to X={tx:.0f} Y={ty:.0f}")
        correction_status = f"point {idx}/{len(CORRECTION_TARGETS)}"

        try:
            move_robot(
                tx, ty, CORRECTION_Z_MM,
                wrist_roll_deg=CORRECTION_WRIST_ROLL_DEG,
                timeout=10,
            )
            time.sleep(CORRECTION_SETTLE_SECONDS)
        except Exception as e:
            print(f"    ✗ IK/bridge failed: {e}")
            continue

        found = None
        pitch_attempts = [None] + list(CORRECTION_WRIST_PITCH_CANDIDATES)
        for pitch in pitch_attempts:
            pitch_label = "IK" if pitch is None else f"{pitch}°"
            correction_status = f"point {idx}/{len(CORRECTION_TARGETS)} pitch {pitch_label}"
            if pitch is not None:
                try:
                    set_servo(2, pitch)
                except Exception as e:
                    print(f"    pitch {pitch_label} ✗ servo failed: {e}")
                    continue
                time.sleep(CORRECTION_SETTLE_SECONDS)

            try:
                true_tip = get_robot_pose_xy()
            except Exception as e:
                print(f"    pitch {pitch_label} ✗ pose read failed: {e}")
                continue

            center, marker_pts, frame, corners, ids = detect_gripper_marker_average()
            if center is not None:
                found = (pitch_label, true_tip, center, marker_pts, frame, corners, ids)
                break
            print(f"    pitch {pitch_label}: marker not visible")

        if found is None:
            print("    ✗ Marker not detected, skipping")
            continue

        detected_pitch, true_tip, center, marker_pts, frame, corners, ids = found

        px, py = float(center[0]), float(center[1])
        observed_marker = pixel_to_robot(px, py, CORRECTION_Z_MM, apply_correction=False)
        if observed_marker is None:
            print("    ✗ Could not convert marker pixel to robot coords")
            continue
        observed_marker = np.array(observed_marker, dtype=np.float64)
        marker_pose = estimate_marker_pose(marker_pts)
        if marker_pose is None:
            print("    ✗ Could not estimate marker pose")
            continue
        R_m2c, t_m2c = marker_pose
        debug_frame = frame

        pitch_suffix = "" if detected_pitch == "IK" else "°"
        print(f"    detected pitch={detected_pitch}{pitch_suffix} marker px=({px:.1f},{py:.1f})")
        print(f"    actual tip=({true_tip[0]:.1f},{true_tip[1]:.1f})")

        samples.append({
            "target_tip_xy": [float(tx), float(ty)],
            "true_tip_xy": [float(true_tip[0]), float(true_tip[1])],
            "marker_pixel": [px, py],
            "observed_marker_xy": [float(observed_marker[0]), float(observed_marker[1])],
            "marker_R_m2c": R_m2c.tolist(),
            "marker_t_m2c": t_m2c.tolist(),
            "detected_wrist_pitch_deg": detected_pitch,
        })

    if len(samples) < 4:
        print(f"  ✗ Only {len(samples)} samples captured. Need at least 4.")
        correction_running = False
        correction_status = "failed"
        return False

    learned_offset_m, learned_rank, learned_errors = solve_marker_tip_offset(samples)
    learned_offset_mm = learned_offset_m * 1000.0
    print()
    print("  Learned marker → tip offset:")
    print(f"    x={learned_offset_mm[0]:+.1f}mm y={learned_offset_mm[1]:+.1f}mm z={learned_offset_mm[2]:+.1f}mm rank={learned_rank}")
    print(f"    length={np.linalg.norm(learned_offset_mm):.1f}mm")

    before_errors = []
    for sample in samples:
        marker_pose = (
            np.array(sample["marker_R_m2c"], dtype=np.float64),
            np.array(sample["marker_t_m2c"], dtype=np.float64),
        )
        observed_tip = marker_pose_to_robot_tip_xy(marker_pose, learned_offset_m)
        true_tip = np.array(sample["true_tip_xy"], dtype=np.float64)
        residual = true_tip - observed_tip
        err = float(np.linalg.norm(residual))
        sample["learned_marker_tip_offset_mm"] = [float(v) for v in learned_offset_mm]
        sample["observed_tip_xy"] = [float(observed_tip[0]), float(observed_tip[1])]
        sample["residual_xy"] = [float(residual[0]), float(residual[1])]
        sample["error_mm"] = err
        before_errors.append(err)

    avg_before = float(np.mean(before_errors))
    max_before = float(np.max(before_errors))

    if avg_before > MAX_REASONABLE_CORRECTION_ERROR_MM:
        print()
        print("  ✗ Correction NOT saved: residuals are too large.")
        print(f"    avg={avg_before:.1f}mm max={max_before:.1f}mm")
        print("    This is probably not the simple 120mm offset anymore. Check:")
        print("    - marker flex: ID21 must not move/bend relative to the gripper")
        print("    - marker pose: ID21 corners must be fully visible and flat")
        print("    - FK/servo model: /pose is model-based, not physical feedback")
        print("    I did not overwrite vision_correction.json.")
        correction_running = False
        correction_status = "failed"
        return False

    after_errors = []
    for i, sample in enumerate(samples):
        corrected = idw_correct(sample["observed_tip_xy"], samples, exclude_index=i)
        true_tip = np.array(sample["true_tip_xy"], dtype=np.float64)
        after_errors.append(float(np.linalg.norm(corrected - true_tip)))

    save_data = {
        "method": "idw_residual_field",
        "marker_dictionary": "DICT_4X4_50",
        "marker_id": GRIPPER_MARKER_ID,
        "marker_tip_distance_mm": MARKER_BEHIND_TIP_MM,
        "learned_marker_tip_offset_mm": [float(v) for v in learned_offset_mm],
        "learned_marker_tip_offset_rank": int(learned_rank),
        "calibration_z_mm": CORRECTION_Z_MM,
        "assumption": "marker-local 3D tip offset is learned automatically from FK pose samples",
        "power": 2.0,
        "max_radius_mm": 220.0,
        "avg_error_before": avg_before,
        "max_error_before": max_before,
        "avg_error_after": float(np.mean(after_errors)),
        "max_error_after": float(np.max(after_errors)),
        "points": samples,
    }

    with open(correction_path, "w") as f:
        json.dump(save_data, f, indent=2)
    if debug_frame is not None:
        cv2.imwrite(correction_debug_path, debug_frame)

    load_vision_correction()
    print()
    print(f"  ✓ Saved correction to {correction_path}")
    print(f"    Before: avg={save_data['avg_error_before']:.1f}mm max={save_data['max_error_before']:.1f}mm")
    print(f"    LOO estimate after: avg={save_data['avg_error_after']:.1f}mm max={save_data['max_error_after']:.1f}mm")
    try:
        requests.get(f"{BRIDGE_URL}/vision/reload", timeout=2)
        print("  ✓ Bridge reloaded vision + correction")
    except Exception:
        print("  ~ Bridge reload skipped; restart or call /vision/reload")
    print("═" * 62)
    correction_running = False
    correction_status = "done"
    return True


def add_manual_correction_point(click_xy, nudge_xy):
    """Append a local correction point from a measured click error."""
    if R_cam2robot is None:
        print("  ✗ Calibrate vision first.")
        return False

    raw_xy = pixel_to_robot(click_xy[0], click_xy[1], PICK_Z, apply_correction=False)
    current_xy = pixel_to_robot(click_xy[0], click_xy[1], PICK_Z, apply_correction=True)
    if raw_xy is None or current_xy is None:
        print("  ✗ Could not convert click to robot coordinates.")
        return False

    raw_xy = np.array(raw_xy, dtype=np.float64)
    current_xy = np.array(current_xy, dtype=np.float64)
    nudge_xy = np.array(nudge_xy, dtype=np.float64)
    desired_xy = current_xy + nudge_xy

    if os.path.exists(correction_path):
        with open(correction_path) as f:
            data = json.load(f)
    else:
        data = {
            "method": "idw_residual_field",
            "power": 2.0,
            "max_radius_mm": 220.0,
            "points": [],
        }

    points = data.setdefault("points", [])
    points.append({
        "source": "manual_click_nudge",
        "target_tip_xy": [float(desired_xy[0]), float(desired_xy[1])],
        "true_tip_xy": [float(desired_xy[0]), float(desired_xy[1])],
        "observed_tip_xy": [float(raw_xy[0]), float(raw_xy[1])],
        "current_corrected_xy": [float(current_xy[0]), float(current_xy[1])],
        "manual_nudge_xy": [float(nudge_xy[0]), float(nudge_xy[1])],
        "marker_pixel": [float(click_xy[0]), float(click_xy[1])],
        "error_mm": float(np.linalg.norm(nudge_xy)),
    })
    data["manual_points"] = int(data.get("manual_points", 0)) + 1
    data["last_manual_nudge_xy"] = [float(nudge_xy[0]), float(nudge_xy[1])]

    with open(correction_path, "w") as f:
        json.dump(data, f, indent=2)

    load_vision_correction()
    try:
        requests.get(f"{BRIDGE_URL}/vision/reload", timeout=2)
        reloaded = True
    except Exception:
        reloaded = False

    print("  ✓ Manual correction point saved")
    print(f"    raw=({raw_xy[0]:.1f},{raw_xy[1]:.1f}) current=({current_xy[0]:.1f},{current_xy[1]:.1f})")
    print(f"    nudge=({nudge_xy[0]:+.1f},{nudge_xy[1]:+.1f}) desired=({desired_xy[0]:.1f},{desired_xy[1]:.1f})")
    print(f"    points={len(points)}{' | bridge reloaded' if reloaded else ' | reload bridge if needed'}")
    return True


def on_mouse(event, x, y, flags, param):
    global last_click
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    last_click = (x, y)

    if mode == "pick":
        result = pixel_to_robot(x, y)
        if result:
            rx, ry = result
            if not is_in_workspace(rx, ry):
                print(f"\n  ✗ ({rx:.0f}, {ry:.0f})mm OUTSIDE workspace")
                return
            if test_mode:
                print(f"\n  [TEST] ({x},{y}) → ({rx:.1f}, {ry:.1f}, {PICK_Z})mm")
            else:
                print(f"\n  Click ({x},{y}) → Robot ({rx:.1f}, {ry:.1f}, {PICK_Z})mm")
                err = send_ik(rx, ry, PICK_Z)
                print(f"    → {'OK err=' + str(round(err, 1)) + 'mm' if err >= 0 else 'FAILED'}")


def stdin_reader():
    while True:
        try:
            input_q.put(input().strip())
        except:
            break


# ─── Load camera calibration ───
if os.path.exists(npz_path):
    npz = np.load(npz_path)
    cam_matrix = npz["camera_matrix"]
    dist_coeffs = npz["dist_coeffs"]
    if "map1" in npz and "map2" in npz:
        remap_map1 = npz["map1"]
        remap_map2 = npz["map2"]
        use_undistort = True
    if "new_camera_matrix" in npz:
        new_cam_matrix = npz["new_camera_matrix"]
    print(f"✓ Camera calibration loaded (undistort={'ON' if use_undistort else 'OFF'})")
elif os.path.exists(calib_path):
    with open(calib_path) as f:
        calib = json.load(f)
    cam_matrix = np.array(calib["camera_matrix"])
    dist_coeffs = np.array(calib["dist_coeffs"])
    if "new_camera_matrix" in calib:
        new_cam_matrix = np.array(calib["new_camera_matrix"])
    print(f"✓ Camera calibration loaded (reproj: {calib.get('reprojection_error', '?')})")
else:
    print("✗ No camera calibration found!")
    print("  Run calibrate_camera.py first.")
    sys.exit(1)

# ─── Setup ChArUco ───
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard(BOARD_SQUARES, SQUARE_SIZE / 1000, MARKER_SIZE / 1000, dictionary)
charuco_detector = cv2.aruco.CharucoDetector(board)

# ─── Load saved transform ───
if os.path.exists(transform_path):
    with open(transform_path) as f:
        saved = json.load(f)
    R_cam2robot = np.array(saved["R_cam2robot"])
    t_cam2robot = np.array(saved["t_cam2robot"])
    table_normal_cam = np.array(saved["table_normal_cam"])
    table_point_cam = np.array(saved["table_point_cam"])
    board_offset_x = saved.get("board_offset_x", 85.0)
    board_offset_y = saved.get("board_offset_y", 0.0)
    axis_mapping = saved.get("axis_mapping", 0)
    PICK_Z = saved.get("pick_z", 100)
    avg_error = saved.get("avg_error")
    mode = "pick"
    print(f"✓ Loaded camera→robot transform (err={avg_error:.2f}mm)")

load_vision_correction()

# ─── Open camera ───
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("✗ Cannot open camera!")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

w = int(cap.get(3))
h = int(cap.get(4))

# Compute undistortion maps if needed
if cam_matrix is not None and not use_undistort:
    if new_cam_matrix is None:
        new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(
            cam_matrix, dist_coeffs, (w, h), alpha=0
        )
    remap_map1, remap_map2 = cv2.initUndistortRectifyMap(
        cam_matrix, dist_coeffs, None, new_cam_matrix, (w, h), cv2.CV_32FC1
    )
    use_undistort = True
    print(f"  ✓ Undistortion maps computed")

print()
print("═" * 60)
print("  IRIS Vision v3 — Board-only solvePnP + Ray-Plane")
print("═" * 60)
print(f"  Camera: {w}x{h} | Undistortion: {'ON ✓' if use_undistort else 'OFF ⚠'}")
print(f"  Board:  {BOARD_SQUARES[0]}x{BOARD_SQUARES[1]}, square={SQUARE_SIZE}mm")
print(f"  Offset: X={board_offset_x}mm  Y={board_offset_y}mm")
print()
print(f"  Workflow: show board fully ({MIN_CHARUCO_CORNERS}+ corners) → SPACE → S")
print("  Click anywhere to get robot coordinates.")
print()
print("  Keys: SPACE=calibrate F=flip O=offset T=test")
print("        G=grid E=errors V=verify S=save +/-=Z R=redo Q=quit")
print("═" * 60)

cv2.namedWindow("IRIS Vision")
cv2.setMouseCallback("IRIS Vision", on_mouse)
threading.Thread(target=stdin_reader, daemon=True).start()

last_charuco_corners = None
last_charuco_ids = None
waiting_for_offset = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Undistort
    if use_undistort:
        frame = cv2.remap(frame, remap_map1, remap_map2, cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display = frame.copy()
    with latest_frame_lock:
        latest_frame = frame.copy()

    # ─── Detect ChArUco ───
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    board_detected = False
    num_corners = 0

    if charuco_corners is not None and len(charuco_corners) >= 4:
        board_detected = True
        num_corners = len(charuco_corners)
        last_charuco_corners = charuco_corners
        last_charuco_ids = charuco_ids

        cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids, (0, 255, 0))
        if marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

        # Show robot coords on corners
        if R_cam2robot is not None:
            cfg = AXIS_CONFIGS[axis_mapping]
            for i in range(0, len(charuco_corners), max(1, len(charuco_corners) // 6)):
                px = charuco_corners[i].flatten()
                result = pixel_to_robot(px[0], px[1])
                if result:
                    rx, ry = result
                    cv2.putText(display, f"({rx:.0f},{ry:.0f})",
                                (int(px[0]) + 8, int(px[1]) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        # ─── Verify mode: show real-time error ───
        # Method: solvePnP gives us the TRUE 3D position of each corner
        # via the board's geometry. pixel_to_robot gives us the PREDICTED
        # position via ray-plane intersection. The difference = system error.
        if verify_mode and R_cam2robot is not None:
            num_cols = BOARD_SQUARES[0] - 1

            # Get true 3D positions via solvePnP on current board detection
            obj_pts_v = []
            for cid in charuco_ids.flatten():
                col = cid % num_cols
                row = cid // num_cols
                obj_pts_v.append([col * SQUARE_SIZE / 1000.0, row * SQUARE_SIZE / 1000.0, 0.0])
            obj_pts_v = np.array(obj_pts_v, dtype=np.float64)
            img_pts_v = charuco_corners.reshape(-1, 2).astype(np.float64)

            solve_cam = new_cam_matrix if use_undistort and new_cam_matrix is not None else cam_matrix
            solve_dist = np.zeros(5) if use_undistort else dist_coeffs

            ok_v, rvec_v, tvec_v = cv2.solvePnP(obj_pts_v, img_pts_v, solve_cam, solve_dist)

            if ok_v:
                R_b2c, _ = cv2.Rodrigues(rvec_v)
                t_b2c = tvec_v.flatten()

                errors_v = []
                for i in range(len(obj_pts_v)):
                    # TRUE position: board→camera→robot (via solvePnP + known transform)
                    pt_cam = R_b2c @ obj_pts_v[i] + t_b2c
                    pt_robot_true = (R_cam2robot @ pt_cam + t_cam2robot) * 1000.0  # mm

                    # PREDICTED position: pixel → ray-plane → robot
                    px = charuco_corners[i].flatten()
                    result = pixel_to_robot(px[0], px[1], apply_correction=False)
                    if result:
                        rx_pred, ry_pred = result
                        err = np.sqrt((rx_pred - pt_robot_true[0]) ** 2 +
                                      (ry_pred - pt_robot_true[1]) ** 2)
                        errors_v.append(err)

                        color = (0, 255, 0) if err < 1.0 else (0, 200, 255) if err < 2.0 else (0, 0, 255)
                        cv2.putText(display, f"{err:.1f}",
                                    (int(px[0]) - 15, int(px[1]) + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

                if errors_v:
                    v_avg = np.mean(errors_v)
                    v_max = np.max(errors_v)
                    cv2.putText(display, f"VERIFY: avg={v_avg:.2f}mm max={v_max:.2f}mm",
                                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # ─── Grid overlay ───
    if show_grid and R_cam2robot is not None:
        grid_step = 50
        # Vertical lines (constant X)
        for rx in range(WORKSPACE_X_MIN, WORKSPACE_X_MAX + 1, grid_step):
            pts_line = []
            for ry in range(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX + 1, 10):
                result = robot_to_pixel(rx, ry)
                if result:
                    px_x, px_y = result
                    if 0 <= px_x < w and 0 <= px_y < h:
                        pts_line.append((px_x, px_y))
            for j in range(len(pts_line) - 1):
                cv2.line(display, pts_line[j], pts_line[j + 1], (50, 50, 50), 1)

        # Horizontal lines (constant Y)
        for ry in range(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX + 1, grid_step):
            pts_line = []
            for rx in range(WORKSPACE_X_MIN, WORKSPACE_X_MAX + 1, 10):
                result = robot_to_pixel(rx, ry)
                if result:
                    px_x, px_y = result
                    if 0 <= px_x < w and 0 <= px_y < h:
                        pts_line.append((px_x, px_y))
            for j in range(len(pts_line) - 1):
                cv2.line(display, pts_line[j], pts_line[j + 1], (50, 50, 50), 1)

        # Workspace boundary
        corners_robot = [
            [WORKSPACE_X_MIN, WORKSPACE_Y_MIN],
            [WORKSPACE_X_MAX, WORKSPACE_Y_MIN],
            [WORKSPACE_X_MAX, WORKSPACE_Y_MAX],
            [WORKSPACE_X_MIN, WORKSPACE_Y_MAX],
        ]
        corners_px = []
        for cx, cy in corners_robot:
            result = robot_to_pixel(cx, cy)
            if result:
                corners_px.append(result)
        if len(corners_px) == 4:
            for j in range(4):
                pt1 = corners_px[j]
                pt2 = corners_px[(j + 1) % 4]
                if all(0 <= c < max(w, h) for c in pt1 + pt2):
                    cv2.line(display, pt1, pt2, (0, 100, 255), 2)

    # ─── Last click ───
    if last_click and mode == "pick":
        cv2.circle(display, last_click, 12, (0, 255, 255), 2)
        cv2.drawMarker(display, last_click, (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        result = pixel_to_robot(last_click[0], last_click[1])
        if result:
            rx, ry = result
            in_ws = is_in_workspace(rx, ry)
            color = (0, 255, 255) if in_ws else (0, 0, 255)
            label = f"({rx:.0f}, {ry:.0f}, {PICK_Z})mm"
            if not in_ws:
                label += " OUT!"
            if test_mode:
                label = "[TEST] " + label
            cv2.putText(display, label,
                        (last_click[0] + 15, last_click[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ─── Header ───
    cv2.rectangle(display, (0, 0), (w, 55), (20, 20, 20), -1)

    if mode == "detect":
        if board_detected:
            ready = num_corners >= MIN_CHARUCO_CORNERS
            txt = f"BOARD | {num_corners}pts/{MIN_CHARUCO_CORNERS} needed | mapping={axis_mapping} | SPACE=calibrate"
            col = (0, 255, 0) if ready else (0, 200, 255)
        else:
            txt = "Show ChArUco board to camera..."
            col = (0, 165, 255)
    else:
        txt = f"PICK | Z={PICK_Z}mm"
        if test_mode:
            txt += " | TEST"
        if verify_mode:
            txt += " | VERIFY"
        if avg_error is not None:
            txt += f" | calib_err={avg_error:.2f}mm"
        if (not BOARD_ONLY_MODE) and vision_correction is not None:
            txt += f" | CORR {len(vision_correction['points'])}pts"
        if correction_running:
            txt += f" | CORRECTING {correction_status}"
        col = (0, 255, 0) if not test_mode else (255, 200, 0)

    cv2.putText(display, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2)

    cfg = AXIS_CONFIGS[axis_mapping]
    method = "solvePnP+RayPlane"
    cv2.putText(display, f"{method} | Offset: X={board_offset_x:.0f} Y={board_offset_y:.0f}mm | {cfg['label']}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)

    # ─── Footer ───
    cv2.rectangle(display, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.putText(display, "SPACE=cal F=flip O=offset T=test G=grid V=verify S=save +/-=Z R=redo Q=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

    cv2.imshow("IRIS Vision", display)
    key = cv2.waitKey(30) & 0xFF

    # ─── Terminal input ───
    try:
        line = input_q.get_nowait()
        if waiting_for_offset:
            parts = line.split()
            if len(parts) >= 1:
                try:
                    board_offset_x = float(parts[0])
                    board_offset_y = float(parts[1]) if len(parts) >= 2 else 0.0
                    print(f"  ✓ Offset: X={board_offset_x}mm Y={board_offset_y}mm")
                    print(f"    Press SPACE to recalibrate")
                    waiting_for_offset = False
                except ValueError:
                    print("  ✗ Type: X Y  (e.g. 85 0)")
        elif waiting_for_manual_correction:
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    dx = float(parts[0])
                    dy = float(parts[1])
                    add_manual_correction_point(manual_correction_anchor, (dx, dy))
                    waiting_for_manual_correction = False
                    manual_correction_anchor = None
                except ValueError:
                    print("  ✗ Type: dx dy  (e.g. -10 0)")
            elif line.lower() in ("q", "cancel", "c"):
                waiting_for_manual_correction = False
                manual_correction_anchor = None
                print("  Manual correction cancelled")
    except queue.Empty:
        pass

    # ─── Keys ───
    if key == ord('q') or key == 27:
        break

    elif key == ord(' '):
        if last_charuco_corners is not None and last_charuco_ids is not None:
            if compute_camera_to_robot(last_charuco_corners, last_charuco_ids):
                mode = "pick"
                print("  → PICK mode! Click anywhere. V=verify with board.")
        else:
            print("  ✗ No board detected")

    elif key == ord('f'):
        axis_mapping = (axis_mapping + 1) % 4
        cfg = AXIS_CONFIGS[axis_mapping]
        print(f"  Axis mapping {axis_mapping}: {cfg['label']}")
        if last_charuco_corners is not None and last_charuco_ids is not None:
            compute_camera_to_robot(last_charuco_corners, last_charuco_ids)
            if R_cam2robot is not None:
                mode = "pick"

    elif key == ord('o'):
        print(f"\n  Current offset: X={board_offset_x}mm Y={board_offset_y}mm")
        print(f"  Type new offset (X Y) in terminal:")
        waiting_for_offset = True

    elif key == ord('t'):
        test_mode = not test_mode
        print(f"  Test mode {'ON' if test_mode else 'OFF'}")

    elif key == ord('g'):
        show_grid = not show_grid
        print(f"  Grid {'ON' if show_grid else 'OFF'}")

    elif key == ord('v'):
        verify_mode = not verify_mode
        print(f"  Verify mode {'ON' if verify_mode else 'OFF'}")
        if verify_mode:
            print("    Move board around workspace — errors shown at each corner")

    elif key == ord('e'):
        show_errors = not show_errors
        print(f"  Error overlay {'ON' if show_errors else 'OFF'}")

    elif key == ord('s'):
        if R_cam2robot is not None:
            if last_charuco_corners is None or len(last_charuco_corners) < MIN_CHARUCO_CORNERS:
                count = 0 if last_charuco_corners is None else len(last_charuco_corners)
                print(f"  ✗ Refusing to save: need {MIN_CHARUCO_CORNERS}+ board corners, have {count}")
                continue
            save_data = {
                "method": "solvePnP_ray_plane",
                "R_cam2robot": R_cam2robot.tolist(),
                "t_cam2robot": t_cam2robot.tolist(),
                "table_normal_cam": table_normal_cam.tolist(),
                "table_point_cam": table_point_cam.tolist(),
                "board_offset_x": board_offset_x,
                "board_offset_y": board_offset_y,
                "axis_mapping": axis_mapping,
                "axis_label": AXIS_CONFIGS[axis_mapping]["label"],
                "pick_z": PICK_Z,
                "avg_error": float(avg_error) if avg_error else None,
                "num_points": int(len(last_charuco_corners)) if last_charuco_corners is not None else 0,
                "undistorted": use_undistort,
                "board_only": True,
                "correction_disabled": True,
                "board_squares": list(BOARD_SQUARES),
                "square_size_mm": SQUARE_SIZE,
                "workspace": {
                    "x_min": WORKSPACE_X_MIN,
                    "x_max": WORKSPACE_X_MAX,
                    "y_min": WORKSPACE_Y_MIN,
                    "y_max": WORKSPACE_Y_MAX,
                },
            }
            with open(transform_path, "w") as f:
                json.dump(save_data, f, indent=2)
            print(f"  ✓ Saved to {transform_path}")
            try:
                requests.get(f"{BRIDGE_URL}/vision/reload", timeout=2)
                print("  ✓ Bridge reloaded vision calibration")
            except Exception:
                print("  ~ Bridge reload skipped; start/reload iris_bridge when ready")
        else:
            print("  ✗ Nothing to save")

    elif key == ord('c'):
        print("  Board-only mode: ID21 correction is disabled. Use SPACE then S with the board fully visible.")

    elif key == ord('m') or key == ord('M'):
        print("  Board-only mode: manual correction is disabled. Fix board offset/axis, then SPACE and S.")

    elif key == ord('r'):
        R_cam2robot = None
        t_cam2robot = None
        table_normal_cam = None
        table_point_cam = None
        mode = "detect"
        last_click = None
        avg_error = None
        print("  Reset — show board and press SPACE")

    elif key == ord('+') or key == ord('='):
        PICK_Z += 10
        print(f"  Z = {PICK_Z}mm")

    elif key == ord('-'):
        PICK_Z = max(10, PICK_Z - 10)
        print(f"  Z = {PICK_Z}mm")

cap.release()
cv2.destroyAllWindows()
