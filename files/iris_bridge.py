#!/usr/bin/env python3
"""
IRIS Serial Bridge v5.0 — ikpy IK + Vision Pick + Gemini Robotics ER
Browser → HTTP → ikpy IK → Serial → ESP32
Vision: ChArUco homography + Gemini Robotics ER 1.6 for object detection
"""

import serial
import json
import time
import threading
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
import os
import subprocess
import sys
import cv2
import base64

# ─── CONFIG ───
SERIAL_PORT = "/dev/tty.usbserial-0001"
BAUD_RATE = 115200
HTTP_PORT = 8765
STEP_SIZE = 1.8
STEP_DELAY = 0.010
CAMERA_ID = 0

# Gemini config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-robotics-er-1.6-preview"

# ─── Platform offset: robot is mounted 23mm above table on PAL board ───
PLATFORM_HEIGHT_MM = 23  # mm — robot base is this much higher than table surface

# ─── Servo config (must match HTML) ───
SERVO_CONFIG = {
    'base_rotation': {'ch': 6, 'offset': 75,  'dir':  1, 'min': 0,   'max': 180},
    'shoulder':      {'ch': 4, 'offset': 160, 'dir': -1, 'min': 30,  'max': 160},
    'elbow':         {'ch': 3, 'offset': 25,  'dir': -1, 'min': 20,  'max': 180},
    'wrist_pitch':   {'ch': 2, 'offset': 100, 'dir': -1, 'min': 0,   'max': 170},
    'wrist_roll':    {'ch': 1, 'offset': 55,  'dir':  1, 'min': 0,   'max': 180},
}
JOINT_NAMES = ['base_rotation', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll']
BASE_Y_CENTER_DEG = 75.0
BASE_Y_INWARD_COMPENSATION = 0.16

# ─── ikpy setup ───
import ikpy.chain

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "iris_arm.urdf")
PS4_MANUAL_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "iris_ps4_manual.py")
PS4_MANUAL_LOG_PATH = os.path.join(SCRIPT_DIR, "iris_ps4_manual.log")
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MEDIA_DIR = os.path.join(SCRIPT_DIR, "media")
DANCE_AUDIO_PATH = os.path.join(MEDIA_DIR, "iris_dance.mp3")
LOCAL_VENV_PYTHON = os.path.join(PROJECT_ROOT, "iris-env", "bin", "python")
PS4_MANUAL_PYTHON = LOCAL_VENV_PYTHON if os.path.exists(LOCAL_VENV_PYTHON) else sys.executable
ps4_manual_process = None
ps4_manual_lock = threading.RLock()
hello_lock = threading.Lock()
hello_running = False
dance_lock = threading.Lock()
dance_running = False
dance_stop_event = threading.Event()

HELLO_POSE_A = {6: 127, 4: 79, 5: 101, 3: 125, 2: 25, 1: 51, 0: 70}
HELLO_POSE_B = {6: 33, 4: 79, 5: 101, 3: 125, 2: 25, 1: 51, 0: 70}
HELLO_WRIST_ROLL_SWING = (28, 78)
HELLO_WAVE_REPETITIONS = 3
DANCE_DEFAULT_BPM = 100.25
DANCE_DEFAULT_DURATION_SECONDS = 30.0
DANCE_MAX_DURATION_SECONDS = 180.0
DANCE_HOME_POSE = {6: 75, 4: 76, 5: 104, 3: 122, 2: 36, 1: 55, 0: 25}
DANCE_MIN_STEP_SIZE = 2.0
DANCE_MAX_STEP_SIZE = 2.25
DANCE_SPEED_PROFILES = {
    "slow": (2.0, 0.018),
    "medium": (2.1, 0.014),
    "fast": (2.2, 0.011),
    "hit": (2.25, 0.010),
}
DANCE_PHASES = [
    {
        "name": "intro_tang_hits",
        "until": 4.0,
        "speed": "slow",
        "steps": [
            ({6: 75, 4: 76, 5: 104, 3: 122, 2: 46, 1: 55, 0: 10}, 1.0),
            ({6: 125, 4: 74, 5: 106, 3: 112, 2: 18, 1: 30, 0: 68}, 1.0),
            ({6: 75, 4: 78, 5: 102, 3: 130, 2: 58, 1: 80, 0: 10}, 1.0),
            ({6: 25, 4: 74, 5: 106, 3: 112, 2: 18, 1: 30, 0: 68}, 1.0),
            ({6: 75, 4: 76, 5: 104, 3: 122, 2: 46, 1: 55, 0: 10}, 0.5),
        ],
    },
    {
        "name": "wide_groove",
        "until": 12.0,
        "speed": "medium",
        "steps": [
            ({6: 130, 4: 72, 5: 108, 3: 108, 2: 66, 1: 22, 0: 14}, 1.0),
            ({6: 20, 4: 88, 5: 92, 3: 148, 2: 12, 1: 98, 0: 70}, 1.0),
            ({6: 122, 4: 70, 5: 110, 3: 104, 2: 70, 1: 34, 0: 20}, 0.5),
            ({6: 28, 4: 90, 5: 90, 3: 152, 2: 10, 1: 86, 0: 64}, 0.5),
            ({6: 75, 4: 62, 5: 118, 3: 108, 2: 72, 1: 55, 0: 12}, 0.5),
            ({6: 75, 4: 88, 5: 92, 3: 148, 2: 16, 1: 55, 0: 66}, 0.5),
        ],
    },
    {
        "name": "beat_change_hits",
        "until": 20.0,
        "speed": "hit",
        "steps": [
            ({6: 75, 4: 42, 5: 138, 3: 92, 2: 78, 1: 55, 0: 6}, 0.25),
            ({6: 75, 4: 88, 5: 92, 3: 154, 2: 12, 1: 55, 0: 70}, 0.25),
            ({6: 132, 4: 58, 5: 122, 3: 96, 2: 74, 1: 14, 0: 10}, 0.25),
            ({6: 18, 4: 90, 5: 90, 3: 156, 2: 10, 1: 104, 0: 68}, 0.25),
            ({6: 126, 4: 68, 5: 112, 3: 110, 2: 58, 1: 28, 0: 14}, 0.5),
            ({6: 24, 4: 88, 5: 92, 3: 150, 2: 16, 1: 92, 0: 64}, 0.5),
        ],
    },
    {
        "name": "wrist_flourish",
        "until": 24.0,
        "speed": "fast",
        "steps": [
            ({6: 120, 4: 66, 5: 114, 3: 106, 2: 64, 1: 8, 0: 16}, 0.25),
            ({6: 120, 4: 66, 5: 114, 3: 106, 2: 64, 1: 55, 0: 64}, 0.25),
            ({6: 30, 4: 88, 5: 92, 3: 150, 2: 14, 1: 108, 0: 16}, 0.25),
            ({6: 30, 4: 88, 5: 92, 3: 150, 2: 14, 1: 55, 0: 66}, 0.25),
            ({6: 130, 4: 70, 5: 110, 3: 112, 2: 58, 1: 22, 0: 22}, 0.5),
            ({6: 20, 4: 88, 5: 92, 3: 150, 2: 16, 1: 98, 0: 58}, 0.5),
        ],
    },
    {
        "name": "finale_sweep",
        "until": 30.0,
        "speed": "fast",
        "steps": [
            ({6: 135, 4: 54, 5: 126, 3: 98, 2: 72, 1: 20, 0: 12}, 0.5),
            ({6: 15, 4: 92, 5: 88, 3: 158, 2: 10, 1: 104, 0: 68}, 0.5),
            ({6: 127, 4: 79, 5: 101, 3: 125, 2: 25, 1: 32, 0: 18}, 0.5),
            ({6: 33, 4: 79, 5: 101, 3: 125, 2: 25, 1: 86, 0: 62}, 0.5),
            ({6: 75, 4: 48, 5: 132, 3: 98, 2: 76, 1: 55, 0: 8}, 0.5),
            ({6: 75, 4: 76, 5: 104, 3: 122, 2: 36, 1: 55, 0: 25}, 0.5),
        ],
    },
]

iris_chain = ikpy.chain.Chain.from_urdf_file(
    URDF_PATH,
    active_links_mask=[False, True, True, True, True, True, False, False]
)
print(f"✓ ikpy chain loaded: {len(iris_chain.links)} links")

# ─── Vision setup ───
CHARUCO_CALIB_PATH = os.path.join(SCRIPT_DIR, "charuco_calibration.json")
VISION_V3_CALIB_PATH = os.path.join(SCRIPT_DIR, "vision_calibration_v3.json")
VISION_CORRECTION_PATH = os.path.join(SCRIPT_DIR, "vision_correction.json")
VISION_DETECTION_DEBUG_PATH = os.path.join(SCRIPT_DIR, "vision_detection_debug.jpg")
CAMERA_CALIB_NPZ_PATH = os.path.join(SCRIPT_DIR, "camera_calibration.npz")
CAMERA_CALIB_JSON_PATH = os.path.join(SCRIPT_DIR, "camera_calibration.json")
USE_VISION_CORRECTION = False
MIN_VISION_CALIB_POINTS = 12
GRIPPER_MARKER_ID = 21
GRIPPER_MARKER_DICT = cv2.aruco.DICT_4X4_50
GRIPPER_MARKER_SIZE_MM = 30.0
CUBE_BBOX_GRASP_Y_FRACTION = 0.50  # V7: visual center first. Do not hide grasp offsets inside detection.
CUBE_GRASP_X_BIAS_MM = 0.0
OBJECT_GRASP_Y_FRACTION = 0.50
OBJECT_GRASP_X_BIAS_MM = 0.0
POINT_GRASP_TOP_OFFSET_PX = 0      # V7: no pixel offset until detection is visibly correct.
POINT_GRASP_MID_OFFSET_PX = 0      # middle workspace: use Gemini's visual center
POINT_GRASP_BOTTOM_OFFSET_PX = 0
POINT_GRASP_TOP_Y_NORM = 0.42
POINT_GRASP_MID_Y_NORM = 0.60
POINT_GRASP_BOTTOM_Y_NORM = 0.82
POINT_GRASP_LEFT_OFFSET_PX = 0
POINT_GRASP_CENTER_X_OFFSET_PX = 0
POINT_GRASP_RIGHT_OFFSET_PX = 0
POINT_GRASP_LEFT_X_NORM = 0.34
POINT_GRASP_CENTER_X_NORM = 0.50
POINT_GRASP_RIGHT_X_NORM = 0.66
WRIST_ROLL_NEUTRAL_DEG = 55.0
WRIST_ROLL_OBJECT_TO_SERVO_SIGN = 1.0
WRIST_ROLL_VISUAL_CORRECTION_SIGN = 1.0
WRIST_ROLL_ALIGNMENT_GAIN = 1.0
WRIST_ROLL_CALIBRATION_BIAS_DEG = 0.0
WRIST_ROLL_MAX_PLANNER_DELTA_DEG = 50.0
WRIST_ROLL_EDGE_MAX_PLANNER_DELTA_DEG = 22.0
WRIST_ROLL_EDGE_MAX_FINAL_DELTA_DEG = 60.0
WRIST_ROLL_EDGE_X_LOW_MM = 190.0
WRIST_ROLL_EDGE_X_HIGH_MM = 340.0
FINAL_ALIGN_CONFIDENCE_THRESHOLD = 0.60
FINAL_ALIGN_MAX_CORRECTIONS = 0
FINAL_ALIGN_MAX_XY_CORRECTION_MM = 40.0
FINAL_ALIGN_MAX_WRIST_DELTA_DEG = 25.0
FINAL_ALIGN_MAX_BASE_DELTA_DEG = 10.0
PRE_GRASP_ROTATE_Z_MM = 70.0
PRE_CLOSE_ALIGN_Z_MM = 35.0
PRE_CLOSE_MAX_Y_CORRECTION_MM = 60.0
PRE_CLOSE_MAX_BASE_DELTA_DEG = 20.0
PRE_CLOSE_MAX_BASE_Y_SHIFT_MM = 85.0
PRE_CLOSE_WRIST_DELTA_GAIN = 1.0
PRE_CLOSE_MAX_WRIST_STEP_DEG = 60.0
PRE_CLOSE_SETTLE_SECONDS = 2.0
PRE_GRASP_WRIST_CONFIDENCE_THRESHOLD = 0.70
PRE_GRASP_MAX_WRIST_DELTA_DEG = 60.0
SIDE_GRASP_WRIST_PITCH_DEG = 27.0
SIDE_GRASP_WRIST_ROLL_DEG = 55.0
SIDE_GRASP_PREAPPROACH_X_BACKOFF_MM = 65.0
SIDE_GRASP_DEFAULT_Z_MM = 60.0
SIDE_GRASP_MIN_Z_MM = 45.0
SIDE_GRASP_MAX_Z_MM = 90.0
SIDE_GRASP_LIFT_Z_MM = 130.0
SIDE_GRASP_TRANSIT_Z_MM = 95.0
vision_homography = None
vision_model = None
vision_correction = None
vision_info = {"loaded": False, "error": None, "avg_error": None, "num_points": 0}

camera_matrix = None
dist_coeffs = None
new_camera_matrix = None
remap_map1 = None
remap_map2 = None
vision_uses_undistorted_pixels = False


def load_camera_calibration():
    """Load camera intrinsics/maps used by v2/v3 vision calibrations."""
    global camera_matrix, dist_coeffs, new_camera_matrix, remap_map1, remap_map2

    if os.path.exists(CAMERA_CALIB_NPZ_PATH):
        data = np.load(CAMERA_CALIB_NPZ_PATH)
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        new_camera_matrix = data["new_camera_matrix"] if "new_camera_matrix" in data else camera_matrix
        remap_map1 = data["map1"] if "map1" in data else None
        remap_map2 = data["map2"] if "map2" in data else None
        print(f"✓ Camera calibration loaded for bridge undistortion")
        return True

    if os.path.exists(CAMERA_CALIB_JSON_PATH):
        with open(CAMERA_CALIB_JSON_PATH) as f:
            data = json.load(f)
        camera_matrix = np.array(data["camera_matrix"])
        dist_coeffs = np.array(data["dist_coeffs"])
        new_camera_matrix = np.array(data.get("new_camera_matrix", data["camera_matrix"]))
        print(f"✓ Camera calibration JSON loaded for bridge undistortion")
        return True

    return False


def ensure_undistortion_maps(width, height):
    global remap_map1, remap_map2, new_camera_matrix
    if camera_matrix is None or dist_coeffs is None:
        return False
    if remap_map1 is not None and remap_map2 is not None:
        if remap_map1.shape[:2] == (height, width):
            return True
    if new_camera_matrix is None:
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (width, height), alpha=0
        )
    remap_map1, remap_map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (width, height), cv2.CV_32FC1
    )
    print(f"✓ Bridge undistortion maps ready for {width}x{height}")
    return True


def load_vision_calibration():
    global vision_homography, vision_info, vision_model, vision_uses_undistorted_pixels

    vision_homography = None
    vision_model = None
    vision_uses_undistorted_pixels = False

    if os.path.exists(VISION_V3_CALIB_PATH):
        try:
            with open(VISION_V3_CALIB_PATH) as f:
                data = json.load(f)
            num_points = int(data.get("num_points", 0))
            if num_points < MIN_VISION_CALIB_POINTS:
                vision_info = {
                    "loaded": False,
                    "error": f"Vision v3 calibration has only {num_points} ChArUco corners; need at least {MIN_VISION_CALIB_POINTS}. Recalibrate with the board fully visible.",
                    "method": "solvePnP_ray_plane",
                    "num_points": num_points,
                }
                print(f"⚠ Vision v3 calibration ignored: only {num_points} corners, need {MIN_VISION_CALIB_POINTS}")
                return
            vision_model = {
                "method": "solvePnP_ray_plane",
                "R_cam2robot": np.array(data["R_cam2robot"], dtype=np.float64),
                "t_cam2robot": np.array(data["t_cam2robot"], dtype=np.float64),
                "table_normal_cam": np.array(data["table_normal_cam"], dtype=np.float64),
                "table_point_cam": np.array(data["table_point_cam"], dtype=np.float64),
            }
            vision_uses_undistorted_pixels = bool(data.get("undistorted", False))
            vision_info = {
                "loaded": True,
                "method": "solvePnP_ray_plane",
                "error": None,
                "avg_error": data.get("avg_error"),
                "num_points": num_points,
                "pick_z": data.get("pick_z", 100),
                "undistorted": vision_uses_undistorted_pixels,
                "workspace": data.get("workspace", {
                    "x_min": 80, "x_max": 350, "y_min": -120, "y_max": 120
                }),
            }
            err = vision_info["avg_error"]
            err_label = f"{err:.2f}mm" if isinstance(err, (int, float)) else "?"
            print(f"✓ Vision v3 calibration loaded (solvePnP+ray-plane, err={err_label})")
            return
        except Exception as e:
            vision_info = {"loaded": False, "error": str(e), "method": "solvePnP_ray_plane"}
            print(f"⚠ Vision v3 calibration error: {e}")

    if os.path.exists(CHARUCO_CALIB_PATH):
        try:
            with open(CHARUCO_CALIB_PATH) as f:
                data = json.load(f)
            vision_homography = np.array(data["homography"])
            vision_uses_undistorted_pixels = bool(data.get("undistorted", False))
            vision_info = {
                "loaded": True,
                "method": "homography",
                "error": None,
                "avg_error": data.get("avg_error"),
                "num_points": data.get("num_points", 0),
                "pick_z": data.get("pick_z", 100),
                "undistorted": vision_uses_undistorted_pixels,
                "workspace": data.get("workspace", {
                    "x_min": 80, "x_max": 350, "y_min": -120, "y_max": 120
                }),
            }
            print(f"✓ Vision calibration loaded ({vision_info['num_points']} pts, err={vision_info['avg_error']:.2f}mm)")
        except Exception as e:
            vision_info = {"loaded": False, "error": str(e)}
            print(f"⚠ Vision calibration error: {e}")
    else:
        print(f"⚠ No vision calibration ({VISION_V3_CALIB_PATH} or {CHARUCO_CALIB_PATH})")


def load_vision_correction():
    global vision_correction
    vision_correction = None
    if not USE_VISION_CORRECTION:
        if vision_info.get("loaded"):
            vision_info["correction_loaded"] = False
            vision_info["correction_disabled"] = True
        print("✓ Vision correction disabled (board-only calibration)")
        return
    if not os.path.exists(VISION_CORRECTION_PATH):
        return
    try:
        with open(VISION_CORRECTION_PATH) as f:
            data = json.load(f)
        points = data.get("points", [])
        if len(points) < 3:
            print(f"⚠ Vision correction ignored: need 3+ points")
            return
        vision_correction = {
            "points": points,
            "power": float(data.get("power", 2.0)),
            "max_radius_mm": float(data.get("max_radius_mm", 220.0)),
            "learned_marker_tip_offset_m": (
                np.array(data["learned_marker_tip_offset_mm"], dtype=np.float64) / 1000.0
                if data.get("learned_marker_tip_offset_mm") is not None else None
            ),
        }
        if vision_info.get("loaded"):
            vision_info["correction_loaded"] = True
            vision_info["correction_points"] = len(points)
            vision_info["correction_avg_error_before"] = data.get("avg_error_before")
            vision_info["correction_avg_error_after"] = data.get("avg_error_after")
        print(f"✓ Vision correction loaded ({len(points)} pts)")
    except Exception as e:
        vision_correction = None
        print(f"⚠ Vision correction error: {e}")


def apply_vision_correction(rx, ry):
    """Apply an inverse-distance weighted residual field in robot XY space."""
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
        dist = (dx * dx + dy * dy) ** 0.5
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
    """Convert pixel coords to robot XY (mm) using v3 ray-plane or legacy homography."""
    if vision_model is not None:
        if camera_matrix is None:
            return None
        K = new_camera_matrix if vision_uses_undistorted_pixels and new_camera_matrix is not None else camera_matrix
        K_inv = np.linalg.inv(K)
        ray_cam = K_inv @ np.array([px, py, 1.0], dtype=np.float64)
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        R_cam2robot = vision_model["R_cam2robot"]
        t_cam2robot = vision_model["t_cam2robot"]
        table_normal_cam = vision_model["table_normal_cam"]
        table_point_cam = vision_model["table_point_cam"]

        denom = float(np.dot(table_normal_cam, ray_cam))
        if abs(denom) < 1e-8:
            return None

        plane_point = table_point_cam
        if z_robot_mm:
            # Move the target plane along robot Z, expressed in camera coordinates.
            z_robot_in_cam = R_cam2robot.T[:, 2]
            plane_point = table_point_cam - z_robot_in_cam * (z_robot_mm / 1000.0)

        ray_t = float(np.dot(table_normal_cam, plane_point) / denom)
        if ray_t < 0:
            return None

        point_cam = ray_t * ray_cam
        point_robot_m = R_cam2robot @ point_cam + t_cam2robot
        rx = float(point_robot_m[0] * 1000.0)
        ry = float(point_robot_m[1] * 1000.0)
        return apply_vision_correction(rx, ry) if apply_correction else (rx, ry)

    if vision_homography is not None:
        h = vision_homography @ [px, py, 1.0]
        rx = float(h[0] / h[2])
        ry = float(h[1] / h[2])
        return apply_vision_correction(rx, ry) if apply_correction else (rx, ry)

    return None


def robot_to_pixel(rx_mm, ry_mm, rz_mm=0.0):
    """Project a robot-frame point to camera pixels."""
    if vision_model is None or camera_matrix is None:
        return None
    K = new_camera_matrix if vision_uses_undistorted_pixels and new_camera_matrix is not None else camera_matrix
    R_cam2robot = vision_model["R_cam2robot"]
    t_cam2robot = vision_model["t_cam2robot"]
    point_robot_m = np.array([rx_mm / 1000.0, ry_mm / 1000.0, rz_mm / 1000.0], dtype=np.float64)
    point_cam = R_cam2robot.T @ (point_robot_m - t_cam2robot)
    if point_cam[2] <= 0:
        return None
    pix = K @ point_cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def estimate_gripper_marker_pose(frame):
    """Detect ID21 and return its marker pose in camera coordinates."""
    if frame is None or camera_matrix is None:
        return None
    dictionary = cv2.aruco.getPredefinedDictionary(GRIPPER_MARKER_DICT)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None

    best_pts = None
    best_area = -1.0
    for idx, marker_id in enumerate(ids.flatten()):
        if int(marker_id) != GRIPPER_MARKER_ID:
            continue
        pts = corners[idx].reshape(4, 2).astype(np.float64)
        area = float(cv2.contourArea(pts.astype(np.float32)))
        if area > best_area:
            best_area = area
            best_pts = pts
    if best_pts is None:
        return None

    s = GRIPPER_MARKER_SIZE_MM / 1000.0
    obj_pts = np.array([
        [-s / 2,  s / 2, 0.0],
        [ s / 2,  s / 2, 0.0],
        [ s / 2, -s / 2, 0.0],
        [-s / 2, -s / 2, 0.0],
    ], dtype=np.float64)
    K = new_camera_matrix if vision_uses_undistorted_pixels and new_camera_matrix is not None else camera_matrix
    solve_dist = np.zeros(5) if vision_uses_undistorted_pixels else dist_coeffs
    ok, rvec, tvec = cv2.solvePnP(obj_pts, best_pts, K, solve_dist)
    if not ok:
        return None
    R_m2c, _ = cv2.Rodrigues(rvec)
    return {
        "R_m2c": R_m2c,
        "t_m2c": tvec.flatten(),
        "marker_pixel": best_pts.mean(axis=0),
    }


def get_visual_gripper_pose():
    """Estimate real gripper-tip pose from ArUco ID21 and learned marker→tip offset."""
    if vision_model is None:
        return {"success": False, "visible": False, "error": "No vision calibration"}
    if not vision_correction or vision_correction.get("learned_marker_tip_offset_m") is None:
        return {"success": False, "visible": False, "error": "No learned marker→tip offset. Run C correction first."}
    frame = get_camera_frame()
    pose = estimate_gripper_marker_pose(frame)
    if pose is None:
        return {"success": True, "visible": False, "error": "ID21 marker not visible"}

    offset_m = vision_correction["learned_marker_tip_offset_m"]
    tip_cam = pose["R_m2c"] @ offset_m + pose["t_m2c"]
    tip_robot = (vision_model["R_cam2robot"] @ tip_cam + vision_model["t_cam2robot"]) * 1000.0

    K = new_camera_matrix if vision_uses_undistorted_pixels and new_camera_matrix is not None else camera_matrix
    tip_px_h = K @ tip_cam
    tip_px = np.array([tip_px_h[0] / tip_px_h[2], tip_px_h[1] / tip_px_h[2]], dtype=np.float64)
    marker_px = pose["marker_pixel"]

    fk_mm, servos, _ = current_fk_pose()
    fk_xy = fk_mm[:2]
    fk_error = float(np.linalg.norm(tip_robot[:2] - fk_xy))

    frame_h, frame_w = frame.shape[:2]
    return {
        "success": True,
        "visible": True,
        "tip": {"x": round(float(tip_robot[0]), 1), "y": round(float(tip_robot[1]), 1), "z": round(float(tip_robot[2]), 1)},
        "tip_pixel": {"x": round(float(tip_px[0]), 1), "y": round(float(tip_px[1]), 1)},
        "tip_norm": {"x": round(float(tip_px[0] / frame_w * 100), 2), "y": round(float(tip_px[1] / frame_h * 100), 2)},
        "marker_pixel": {"x": round(float(marker_px[0]), 1), "y": round(float(marker_px[1]), 1)},
        "marker_norm": {"x": round(float(marker_px[0] / frame_w * 100), 2), "y": round(float(marker_px[1] / frame_h * 100), 2)},
        "fk_tip": {"x": round(float(fk_xy[0]), 1), "y": round(float(fk_xy[1]), 1)},
        "fk_error_mm": round(fk_error, 1),
        "servos": {str(ch): round(float(angle), 1) for ch, angle in servos.items()},
    }


def check_target_alignment(target_x, target_y, target_z=0.0, tolerance_mm=5.0):
    """Compare target XY to visual gripper-tip XY and return a numeric correction."""
    pose = get_visual_gripper_pose()
    target_px = robot_to_pixel(float(target_x), float(target_y), float(target_z))
    frame = get_camera_frame()
    if target_px is not None and frame is not None:
        frame_h, frame_w = frame.shape[:2]
        pose["target_pixel"] = {"x": round(float(target_px[0]), 1), "y": round(float(target_px[1]), 1)}
        pose["target_norm"] = {"x": round(float(target_px[0] / frame_w * 100), 2), "y": round(float(target_px[1] / frame_h * 100), 2)}

    if not pose.get("visible"):
        pose.update({
            "aligned": False,
            "error_mm": None,
            "dx_mm": 0,
            "dy_mm": 0,
            "suggested_dx_mm": 0,
            "suggested_dy_mm": 0,
        })
        return pose

    tip = pose["tip"]
    dx = float(target_x) - float(tip["x"])
    dy = float(target_y) - float(tip["y"])
    err = float((dx * dx + dy * dy) ** 0.5)
    max_step = 12.0
    pose.update({
        "target": {"x": round(float(target_x), 1), "y": round(float(target_y), 1), "z": round(float(target_z), 1)},
        "aligned": err <= float(tolerance_mm),
        "error_mm": round(err, 1),
        "dx_mm": round(dx, 1),
        "dy_mm": round(dy, 1),
        "suggested_dx_mm": round(float(np.clip(dx, -max_step, max_step)), 1),
        "suggested_dy_mm": round(float(np.clip(dy, -max_step, max_step)), 1),
        "tolerance_mm": float(tolerance_mm),
    })
    return pose


load_camera_calibration()
load_vision_calibration()
load_vision_correction()

# ─── Gemini Robotics ER ───
def query_gemini(prompt, image_bytes):
    """Send image + prompt to Gemini Robotics ER, return parsed JSON points."""
    import urllib.request

    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set. Export it as environment variable."}

    img_b64 = base64.b64encode(image_bytes).decode('utf-8')

    request_body = {
        "contents": [{
            "parts": [
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": img_b64
                    }
                },
                {
                    "text": prompt
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.5,
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    req = urllib.request.Request(
        url,
        data=json.dumps(request_body).encode('utf-8'),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))

        # Extract text from response
        text = ""
        for candidate in result.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    text += part["text"]

        # Try to parse JSON from response
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()

        try:
            points = json.loads(clean)
            return {"ok": True, "points": points, "raw": text}
        except json.JSONDecodeError:
            return {"ok": True, "points": [], "raw": text}

    except Exception as e:
        return {"error": str(e)}

def gemini_points_to_pixels(points, img_width, img_height):
    """Convert Gemini normalized [y,x] 0-1000 points to pixel coordinates."""
    result = []
    for p in points:
        if "point" in p:
            ny, nx = p["point"]
            px = int(nx * img_width / 1000)
            py = int(ny * img_height / 1000)
            result.append({
                "px": px, "py": py,
                "label": p.get("label", ""),
                "norm": p["point"]
            })
    return result

if GEMINI_API_KEY:
    print(f"✓ Gemini API key loaded ({GEMINI_MODEL})")
else:
    print(f"⚠ No GEMINI_API_KEY — set with: export GEMINI_API_KEY=your_key")

# ─── AGENT: System Prompt + Tool Declarations + Executor ───
AGENT_SYSTEM_PROMPT = """You are IRIS — an intelligent robot arm assistant. You control a 6-DOF robot arm with a gripper through function calls.
You see through a camera mounted above the workspace.

═══ REACH / COORDINATES ═══
- X axis: forward from base. Y axis: negative=right, positive=left.
- Do NOT reject objects because they look outside an old workspace rectangle.
- The true limit is the robot arm IK. If move_arm returns success=false or high error_mm, the target is not reachable enough.
- Z axis: -15 to 250mm. Z=200 safe transit height.
  • Z=0mm — default grab height for objects on table
  • Z=-10mm — first retry height
  • Z=-20mm — second retry height
  • Z=20-40mm — for taller objects

═══ WRIST ROLL (GRIPPER ROTATION) ═══
wrist_roll_deg controls gripper jaw rotation (0-180°).
At 55° the gripper is STRAIGHT — aligned with the line from robot base to camera center.
wrist_pitch_deg controls tool pitch. For side grasps, use the bridge side-grasp tools instead of guessing pitch manually.

CRITICAL: The gripper is SMALL (~4cm). ALWAYS grab objects across their NARROWEST side!
- Remote control → rotate so jaws close on the SHORT width, NOT along the length
- Pen/marker → jaws perpendicular to the pen
- Cubes → if the cube is rotated/tilted, align wrist roll to match its edges!
  A cube sitting at an angle still needs wrist roll adjustment to grab it cleanly.
- Bottles/test tubes/cups/cylinders → do NOT top-grab. Use side_grasp_object so the gripper approaches the middle from the side with the gripper roughly parallel to the table.

When to rotate: if the object (ANY object, including cubes) is not straight relative to the gripper.
Do NOT rotate wrist roll during the first approach. Keep wrist roll neutral until the gripper is low above the object.
detect_objects returns wrist_roll_suggestion_deg only as debug/diagnostic information. Do not use it for the initial move.
Final wrist roll is chosen only by align_before_descent, where the gripper and object are visible together.

═══ ORCHESTRATION PHILOSOPHY ═══
You are the planner. Do NOT use a fixed one-shot routine.
Observe → move → observe again → adjust → grab.
The camera and robot have small errors, so your job is to close the loop visually before closing the gripper.

STEP 1 — DETECT: detect_objects("object name") → get X, Y
  → Use one detection first. The tool returns the visual center of the object.
  → Use the returned X/Y directly; do not add offsets.
  → Ignore wrist_roll_suggestion_deg for motion. It is only debug.
  → If detect_objects returns bbox_grasp_candidates, they are precomputed from the detector box: left_inset, center, right_inset.
  → Prefer those candidates when the center looks awkward. Do not invent arbitrary extra points; choose one of the bbox candidates.
  → If a grasp fails and bbox_grasp_candidates exist, retry a different candidate from the same box before inventing offsets:
    first center, then left_inset or right_inset depending on which side looks better.
STEP 2 — PREPARE: open_gripper → move_to_safe_height → wait_seconds(1.5)
STEP 3 — ABOVE TARGET: move_arm(target_x, target_y, 200) → wait_seconds(1.5)
STEP 4 — PRE-GRASP WRIST CHECK:
  1. move_arm(target_x, target_y, 70, wrist_roll_deg=55) → wait_seconds(2)
  2. call align_before_descent(object, target_x, target_y, 55)
  3. Use the returned target_y and wrist_roll_deg for final descent.
  → align_before_descent lowers to about Z=35 first, then corrects wrist roll and Y left/right.
  → Base/Y trim uses degrees: positive/increase = gripper moves LEFT, negative/decrease = gripper moves RIGHT.
  → wrist_roll positive servo degrees rotate clockwise in the camera image; negative rotates counter-clockwise.
STEP 5 — DESCEND: move_arm(corrected_target_x, corrected_target_y, 0, wrist_roll_deg=corrected_wrist_roll_deg) → wait_seconds(2.5)
STEP 6 — VISUAL ALIGNMENT:
  1. For now, do not use check_target_alignment unless explicitly asked.
  2. Before closing, call align_gripper_for_close(object, target_x, target_y, target_z, wrist_roll_deg).
  → If it returns ready_to_close=true, then close_gripper.
  → If it returns requires_redetect=true, detect_objects again and retry from STEP 3.
  → DON'T OVERTHINK IT: if the object is visibly between the jaws, close_gripper. Do not chase tiny cosmetic errors.
  → If confidence is medium but the object looks inside the gripper, accept it and grab. The robot is allowed to be practical.
  → Do not perform X/Y/wrist corrections here. Corrections belong in align_before_descent before the final descent.
  → Final alignment is only a sanity check: close if likely good, otherwise lift and re-detect.
STEP 7 — GRAB: close_gripper → wait_seconds(1.5)
STEP 8 — LIFT: move_arm(target_x, target_y, 80) → wait_seconds(2) → check_success("holding object")
  → If FAILED (max 2 retries):
    1. open_gripper → wait_seconds(1)
    2. detect_objects AGAIN (object may have moved!)
    3. If bbox_grasp_candidates exist, try another candidate from the box. Otherwise retry with NEW X,Y coords but Z 10mm LOWER (-10, then -20)
    4. Keep wrist_roll neutral until align_before_descent runs again.
    5. After 2 failed retries → go_home, tell user
STEP 9 — TRANSPORT: move_to_safe_height → wait_seconds(2) → move_arm(dest_x, dest_y, 200) → wait_seconds(2)
STEP 10 — PLACE: move_arm(dest_x, dest_y, 30) → wait_seconds(2)
  → Before open_gripper, use check_target_alignment(dest_x, dest_y, 30) if ID21 is visible.
  → Adjust up to 4 times with 10-18mm X/Y moves, then open_gripper.
STEP 11 — DONE: move_to_safe_height → go_home

SIDE GRASP:
- For a bottle, test tube, cup, cylinder, or tall container: call side_grasp_object("object").
- It detects the object, opens the gripper, approaches the object at side-grasp height with wrist pitch set for a side grasp, closes, and lifts vertically.
- side_grasp_object already returns holding_object=true and lifted=true when it succeeds. Do not stop after only asking whether it holds the object if the user requested more steps.
- After a successful side grasp, keep the returned wrist_pitch_deg and wrist_roll_deg during transport moves. Do not let IK freely reset the wrist while holding a bottle/test tube.

═══ CRITICAL RULES ═══
0. You must orchestrate using the tools. Think visually and adapt after each observation.
1. ALWAYS wait_seconds after move_arm. 2-3s for big moves.
2. ALWAYS open_gripper BEFORE descending.
3. ALWAYS move to Z=200 before moving to new X,Y.
4. ALWAYS detect_objects FIRST. Never guess positions.
5. Do not close_gripper until align_gripper_for_close returns ready_to_close=true.
6. On retry, ALWAYS detect_objects again for new X/Y, but do not rotate wrist until align_before_descent.
7. Maximum 2 retries then stop.
8. The preferred wrist rotation moment is pre-grasp height around Z=70, not during the final descent.
9. check_success is only a checkpoint. If completed=true and the user's request has more steps, continue the task.
10. If an object is not detected after retries, STOP. Do not keep moving around an empty table spot.

═══ PERSONALITY ═══
- Speak Romanian, be concise
- If task fails after retries, tell user honestly
- Move quickly but precisely"""

AGENT_TOOL_DECLARATIONS = [
    {
        "name": "move_arm",
        "description": "Move the robot arm to an XYZ position in millimeters using inverse kinematics. During top picking, keep wrist_roll_deg neutral at 55 until align_before_descent. For side grasps, prefer side_grasp_object.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Forward distance from base in mm. Reachability is checked by IK result/error."},
                "y": {"type": "number", "description": "Left/right offset in mm, negative=right, positive=left. Reachability is checked by IK result/error."},
                "z": {"type": "number", "description": "Height in mm (-10 to 250. -10 to 0=below table level for flat objects, 0=table, 200=safe transit)"},
                "wrist_roll_deg": {"type": "number", "description": "Wrist roll servo angle (0-180). For top picks, use 55 during approach. Do not use detect_objects wrist_roll_suggestion_deg until align_before_descent decides visually."},
                "wrist_pitch_deg": {"type": "number", "description": "Optional wrist pitch servo angle (0-170). Use only for deliberate tool orientation; for bottles/cylinders prefer side_grasp_object."}
            },
            "required": ["x", "y", "z"]
        }
    },
    {
        "name": "open_gripper",
        "description": "Open the gripper to release an object.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "close_gripper",
        "description": "Close the gripper to grab an object.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "wave_hello",
        "description": "Run the friendly IRIS audience greeting/wave routine.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "move_to_safe_height",
        "description": "Move arm to safe transit height (Z=200mm).",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "go_home",
        "description": "Move arm to home/rest position. Opens gripper and centers arm raised.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "detect_objects",
        "description": "Use camera + Gemini ER vision plus local contour refinement to find objects on the table. Returns a final XYZ grasp target, debug wrist_roll_suggestion_deg, and bbox_grasp_candidates when a detector box is available.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to detect, e.g. 'red cube', 'all objects'"},
                "samples": {"type": "number", "description": "Usually leave at 1. Use 3 only if the first detection is visibly unstable."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "check_success",
        "description": "Take a photo and verify if a task was completed.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_description": {"type": "string", "description": "What should have happened"}
            },
            "required": ["task_description"]
        }
    },
    {
        "name": "check_gripper_proximity",
        "description": "Look with the camera after the gripper has descended near an object. Returns whether the object is centered between the jaws and suggests small X/Y/wrist adjustments. Use before close_gripper.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_description": {"type": "string", "description": "The object being grabbed, e.g. 'red cube' or 'remote control'"}
            },
            "required": ["object_description"]
        }
    },
    {
        "name": "check_wrist_roll_alignment",
        "description": "At pre-grasp height around Z=70, look at the gripper above/near the object and judge only wrist-roll orientation. Returns a small wrist_roll_delta_deg to align the jaws before final descent.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_description": {"type": "string", "description": "The object being grabbed, e.g. '2x2 cube'."},
                "current_wrist_roll_deg": {"type": "number", "description": "Current wrist roll servo angle."}
            },
            "required": ["object_description", "current_wrist_roll_deg"]
        }
    },
    {
        "name": "align_before_descent",
        "description": "Lower near the object before the final drop, then visually correct wrist roll and Y left/right. Positive dy_mm moves left, negative moves right. Positive base_servo_delta_deg increases base degrees and moves left; negative moves right. Use returned target_y and wrist_roll_deg for final descent.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_description": {"type": "string", "description": "The object being grabbed, e.g. '2x2 cube', 'cup', or 'test tube'."},
                "target_x": {"type": "number", "description": "Current grasp target X in robot mm."},
                "target_y": {"type": "number", "description": "Current grasp target Y in robot mm."},
                "current_wrist_roll_deg": {"type": "number", "description": "Current wrist roll servo angle."},
                "target_z": {"type": "number", "description": "Low pre-close alignment height, default 35mm."}
            },
            "required": ["object_description", "target_x", "target_y", "current_wrist_roll_deg"]
        }
    },
    {
        "name": "align_gripper_for_close",
        "description": "Final closed-loop guard before close_gripper. It checks whether the object is inside the jaws, applies at most 2 corrections, and returns ready_to_close=true only when it is safe to close. If unsure, it opens/lifts and asks for re-detection.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_description": {"type": "string", "description": "The object being grabbed, e.g. 'red cube' or '3x3 cube'."},
                "target_x": {"type": "number", "description": "Current grasp target X in robot mm."},
                "target_y": {"type": "number", "description": "Current grasp target Y in robot mm."},
                "target_z": {"type": "number", "description": "Current grasp target Z in robot mm, usually 0."},
                "wrist_roll_deg": {"type": "number", "description": "Current wrist roll angle. For top picks this is usually the value returned by align_before_descent, not detect_objects."}
            },
            "required": ["object_description", "target_x", "target_y"]
        }
    },
    {
        "name": "get_gripper_pose",
        "description": "Estimate the real visual gripper-tip position using ArUco ID21 on the gripper. Returns tip X/Y, marker visibility, pixel position, and FK-vs-vision error.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "check_target_alignment",
        "description": "Compare a target robot X/Y against the real gripper tip from ID21 and return numeric correction dx/dy. Use this after descending near a target and before close_gripper.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_x": {"type": "number", "description": "Target object X in robot mm"},
                "target_y": {"type": "number", "description": "Target object Y in robot mm"},
                "target_z": {"type": "number", "description": "Target height in mm, usually grab height or object height"},
                "tolerance_mm": {"type": "number", "description": "Acceptable XY error, default 5"}
            },
            "required": ["target_x", "target_y"]
        }
    },
    {
        "name": "side_grasp_object",
        "description": "Detect and side-grasp a tall object such as a bottle, test tube, cup, or cylinder. The gripper approaches the middle of the object from the side with wrist pitch set for a table-parallel side grasp, then closes and lifts.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_description": {"type": "string", "description": "Object to side-grasp, e.g. 'small bottle', 'test tube', 'cup'."},
                "grasp_z": {"type": "number", "description": "Optional side grasp height in mm. Values are clamped to a safe side range; omit this unless you have a strong reason."},
                "lift_z": {"type": "number", "description": "Optional lift height after closing, default 130mm."}
            },
            "required": ["object_description"]
        }
    },
    {
        "name": "set_servo_direct",
        "description": "Directly set one servo to an absolute angle. Prefer trim_base_rotation for lateral fine alignment — that one uses a small delta from the current angle, no need to know the absolute value.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {"type": "number", "description": "Servo channel."},
                "angle": {"type": "number", "description": "Absolute servo angle in degrees."}
            },
            "required": ["channel", "angle"]
        }
    },
    {
        "name": "trim_base_rotation",
        "description": "Apply a small DELTA to the current base rotation servo (channel 6) for final lateral alignment. There is no fixed neutral — the delta is added to wherever the base servo is right now. Positive/increase moves the gripper LEFT; negative/decrease moves the gripper RIGHT. Use small trims only.",
        "parameters": {
            "type": "object",
            "properties": {
                "delta_deg": {"type": "number", "description": "Degrees to add to the current base servo angle. Positive = gripper left, negative = gripper right. Keep small (+/-3 to +/-7, max +/-10)."}
            },
            "required": ["delta_deg"]
        }
    },
    {
        "name": "wait_seconds",
        "description": "Wait for servos to finish moving.",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "Seconds to wait (0.1-10)"}
            },
            "required": ["seconds"]
        }
    },
    {
        "name": "describe_scene",
        "description": "Take a photo and describe everything visible on the table.",
        "parameters": {"type": "object", "properties": {}}
    },
]

held_object_state = {
    "holding": False,
    "grasp_mode": None,
    "object_description": None,
    "wrist_pitch_deg": None,
    "wrist_roll_deg": None,
}


def move_arm_direct(x, y, z, wrist_roll_deg=None, wrist_pitch_deg=None):
    if held_object_state.get("holding") and held_object_state.get("grasp_mode") == "side":
        if wrist_roll_deg is None:
            wrist_roll_deg = held_object_state.get("wrist_roll_deg")
        if wrist_pitch_deg is None:
            wrist_pitch_deg = held_object_state.get("wrist_pitch_deg")
    # Agent/Live commands use table-relative Z, same as /ik and Vision Pick.
    z_adjusted = float(z) - PLATFORM_HEIGHT_MM
    target_m = [float(x) / 1000, float(y) / 1000, z_adjusted / 1000]
    servo_angles, error, actual_mm = solve_ik(target_m, wrist_pitch_deg=wrist_pitch_deg)
    if error > 35:
        print(f"  Agent → Arm target unreachable-ish ({x},{y},{z})mm err={error:.1f}mm; not moving")
        return {
            "success": False,
            "error": "IK error too large; target is outside reliable arm reach",
            "error_mm": round(error, 1),
            "actual": {"x": round(actual_mm[0], 1), "y": round(actual_mm[1], 1), "z": round(actual_mm[2], 1)},
        }
    wp_cfg = SERVO_CONFIG['wrist_pitch']
    wp_angle = servo_angles[wp_cfg['ch']]

    wr_cfg = SERVO_CONFIG['wrist_roll']
    if wrist_roll_deg is not None:
        wr_angle = int(round(np.clip(wrist_roll_deg, wr_cfg['min'], wr_cfg['max'])))
        last_wrist_roll[0] = wr_angle
    else:
        wr_angle = last_wrist_roll[0]
    servo_angles[wr_cfg['ch']] = wr_angle
    servo_angles[5] = 180 - servo_angles[4]
    sent_servo_angles = compensated_servo_targets(servo_angles)
    for ch, angle in servo_angles.items():
        set_target(ch, angle)
    print(f"  Agent → Arm ({actual_mm[0]:.0f},{actual_mm[1]:.0f},{actual_mm[2]:.0f})mm err={error:.1f}mm wp={wp_angle}° wr={wr_angle}° servos={sent_servo_angles}")
    return {
        "success": True,
        "actual": {"x": round(actual_mm[0], 1), "y": round(actual_mm[1], 1), "z": round(actual_mm[2], 1)},
        "error_mm": round(error, 1),
        "wrist_pitch": wp_angle,
        "wrist_roll": wr_angle,
        "servos": sent_servo_angles,
        "ik_servos_raw": servo_angles,
    }


def set_gripper(opened):
    angle = 0 if opened else 70
    send_raw(0, angle)
    with move_lock:
        current_angles[0] = angle
        target_angles[0] = angle
    if opened:
        held_object_state.update({
            "holding": False,
            "grasp_mode": None,
            "object_description": None,
            "wrist_pitch_deg": None,
            "wrist_roll_deg": None,
        })
    return {"success": True, "state": "open" if opened else "closed"}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def y_target_from_base_delta(x_mm, y_mm, base_delta_deg):
    """Approximate a small base-servo trim as a lateral Y target shift."""
    x_mm = _safe_float(x_mm)
    y_mm = _safe_float(y_mm)
    base_delta_deg = _safe_float(base_delta_deg)
    dy = x_mm * np.tan(np.radians(base_delta_deg))
    dy = float(np.clip(dy, -PRE_CLOSE_MAX_BASE_Y_SHIFT_MM, PRE_CLOSE_MAX_BASE_Y_SHIFT_MM))
    return y_mm + dy, dy


def align_gripper_for_close_direct(object_description, target_x, target_y, target_z=0.0,
                                   wrist_roll_deg=None, max_corrections=FINAL_ALIGN_MAX_CORRECTIONS):
    """Closed-loop final alignment guard. It never closes the gripper itself."""
    current_x = _safe_float(target_x)
    current_y = _safe_float(target_y)
    current_z = _safe_float(target_z)
    current_wr = _safe_float(wrist_roll_deg, last_wrist_roll[0]) if wrist_roll_deg is not None else last_wrist_roll[0]
    max_corrections = int(max(0, min(3, max_corrections)))
    history = []

    for attempt in range(max_corrections + 1):
        check = agent_execute_function("check_gripper_proximity", {
            "object_description": object_description,
        })
        history.append(check)
        confidence = _safe_float(check.get("confidence", 0.0), 0.0)
        if check.get("success") and check.get("aligned") and confidence >= FINAL_ALIGN_CONFIDENCE_THRESHOLD:
            return {
                "success": True,
                "ready_to_close": True,
                "attempts": attempt,
                "confidence": round(confidence, 2),
                "target": {"x": round(current_x, 1), "y": round(current_y, 1), "z": round(current_z, 1)},
                "wrist_roll_deg": round(current_wr, 1),
                "history": history,
            }

        if attempt >= max_corrections:
            break

        # Final close alignment is only a sanity check. Do not move here:
        # late corrections tend to overthink good grasps and knock objects away.
        break

    set_gripper(True)
    move_arm_direct(current_x, current_y, 200.0, current_wr)
    return {
        "success": True,
        "ready_to_close": False,
        "requires_redetect": True,
        "reason": "Final visual alignment is not confident enough; gripper opened and lifted to safe height.",
        "target": {"x": round(current_x, 1), "y": round(current_y, 1), "z": round(current_z, 1)},
        "wrist_roll_deg": round(current_wr, 1),
        "history": history,
    }


def check_wrist_roll_alignment_direct(object_description, current_wrist_roll_deg):
    """Ask the vision model for a small wrist-roll correction at pre-grasp height."""
    import urllib.request as urlreq

    frame = get_camera_frame()
    if frame is None:
        return {"success": False, "error": "Camera not running"}
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    current_wr = _safe_float(current_wrist_roll_deg, last_wrist_roll[0])

    prompt = f"""Look at this robot camera image. The gripper is near/above the object before the final descent.
Object: "{object_description}"
Current wrist roll servo angle: {current_wr:.1f} degrees.

Judge ONLY wrist roll orientation: are the gripper jaws aligned with the object's grasp direction?
For a cube, compare the gripper jaw line to the cube top-face edges. If the cube is rotated about 45 degrees, the wrist correction should be modest, around that kind of angle, not a huge swing.
Do not suggest X/Y/Z movement here. Only wrist roll.

Use image-space signs only:
- positive wrist_roll_delta_deg = the gripper should rotate clockwise in the camera image.
- negative wrist_roll_delta_deg = the gripper should rotate counter-clockwise in the camera image.
If unsure, use a small delta or 0 and low confidence.
Never suggest more than +/-20 degrees. Most cup/cube corrections should be around 5-15 degrees, but use more if clearly needed.

Respond ONLY in JSON:
{{
  "aligned": true/false,
  "confidence": 0.0-1.0,
  "wrist_roll_delta_deg": 0,
  "reasoning": "brief Romanian explanation"
}}"""

    request_body = {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
            {"text": prompt}
        ]}],
        "generationConfig": {"temperature": 0.15, "thinkingConfig": {"thinkingBudget": 512}}
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    req = urlreq.Request(url, data=json.dumps(request_body).encode('utf-8'),
                         headers={"Content-Type": "application/json"}, method="POST")
    with urlreq.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read().decode('utf-8'))

    text = ""
    for candidate in result.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                text += part["text"]
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        clean = clean.rsplit("```", 1)[0]
    try:
        check = json.loads(clean.strip())
    except Exception:
        return {"success": True, "aligned": False, "confidence": 0.0, "wrist_roll_delta_deg": 0.0, "raw_response": text[:240]}

    raw_delta = float(np.clip(
        _safe_float(check.get("wrist_roll_delta_deg", 0.0), 0.0),
        -PRE_GRASP_MAX_WRIST_DELTA_DEG,
        PRE_GRASP_MAX_WRIST_DELTA_DEG,
    ))
    delta = float(np.clip(
        raw_delta * WRIST_ROLL_VISUAL_CORRECTION_SIGN,
        -PRE_GRASP_MAX_WRIST_DELTA_DEG,
        PRE_GRASP_MAX_WRIST_DELTA_DEG,
    ))
    confidence = _safe_float(check.get("confidence", 0.0), 0.0)
    aligned = bool(check.get("aligned")) and confidence >= PRE_GRASP_WRIST_CONFIDENCE_THRESHOLD
    return {
        "success": True,
        "aligned": aligned,
        "confidence": round(confidence, 2),
        "wrist_roll_delta_deg": round(delta, 1),
        "raw_image_wrist_roll_delta_deg": round(raw_delta, 1),
        "visual_correction_sign": WRIST_ROLL_VISUAL_CORRECTION_SIGN,
        "suggested_wrist_roll_deg": round(float(np.clip(
            current_wr + delta,
            SERVO_CONFIG["wrist_roll"]["min"],
            SERVO_CONFIG["wrist_roll"]["max"],
        )), 1),
        "reasoning": check.get("reasoning", ""),
    }


def align_before_descent_direct(object_description, target_x, target_y, current_wrist_roll_deg,
                                target_z=PRE_CLOSE_ALIGN_Z_MM, max_corrections=2):
    """Low pre-close visual alignment. Applies wrist-roll and small base/Y corrections."""
    import urllib.request as urlreq

    current_x = _safe_float(target_x)
    current_y = _safe_float(target_y)
    current_z = _safe_float(target_z, PRE_CLOSE_ALIGN_Z_MM)
    # For top grasps, ignore detector/planner wrist suggestions. Start from a
    # known neutral pose and let the low visual check choose the roll.
    current_wr = WRIST_ROLL_NEUTRAL_DEG
    current_wr = clamp_wrist_roll_for_workspace(current_wr, current_x)
    max_corrections = int(max(0, min(3, max_corrections)))
    history = []

    move_arm_direct(current_x, current_y, current_z, current_wr)
    time.sleep(PRE_CLOSE_SETTLE_SECONDS)

    for attempt in range(max_corrections + 1):
        frame = get_camera_frame()
        if frame is None:
            return {"success": False, "error": "Camera not running"}
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        prompt = f"""Look at this robot camera image. The gripper is LOW and close to the object, just before the final descent and close.
Object: "{object_description}"
Current target: x={current_x:.1f}mm, y={current_y:.1f}mm, z={current_z:.1f}mm
Current wrist roll servo angle: {current_wr:.1f} degrees.

Goal: judge wrist roll and Y left-right alignment at this low pre-close height.
Do not suggest X movement. Use dy_mm for left/right. You may use base_servo_delta_deg only as an optional yaw trim.
wrist_roll_delta_deg is an image-space correction, usually -15..35, max +/-60.
dy_mm is a robot Y correction, usually -15..25, max +/-60.
base_servo_delta_deg is a small base rotation trim, usually -10..10, max +/-20.
This is the ONLY wrist-roll decision point for top picks, so use enough correction when clearly needed.

For wrist_roll_delta_deg signs:
- positive = increase servo degrees; this rotates the gripper clockwise in the camera image
- negative = decrease servo degrees; this rotates the gripper counter-clockwise in the camera image
For base_servo_delta_deg signs:
- positive = increase base servo degrees; gripper moves LEFT in the camera/workspace
- negative = decrease base servo degrees; gripper moves RIGHT in the camera/workspace
For dy_mm signs:
- positive = move gripper LEFT
- negative = move gripper RIGHT

If it is already good enough, aligned=true and deltas 0.
Respond ONLY in JSON:
{{
  "aligned": true/false,
  "confidence": 0.0-1.0,
  "dy_mm": 0,
  "wrist_roll_delta_deg": 0,
  "base_servo_delta_deg": 0,
  "reasoning": "brief Romanian explanation"
}}"""

        request_body = {
            "contents": [{"parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                {"text": prompt}
            ]}],
            "generationConfig": {"temperature": 0.2, "thinkingConfig": {"thinkingBudget": 768}}
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        req = urlreq.Request(url, data=json.dumps(request_body).encode('utf-8'),
                             headers={"Content-Type": "application/json"}, method="POST")
        with urlreq.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode('utf-8'))

        text = ""
        for candidate in result.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    text += part["text"]
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            clean = clean.rsplit("```", 1)[0]
        try:
            check = json.loads(clean.strip())
        except Exception:
            check = {"aligned": False, "confidence": 0.0, "dy_mm": 0, "wrist_roll_delta_deg": 0, "base_servo_delta_deg": 0, "reasoning": text[:200]}

        confidence = _safe_float(check.get("confidence", 0.0), 0.0)
        raw_wr_delta = float(np.clip(
            _safe_float(check.get("wrist_roll_delta_deg", 0.0), 0.0),
            -PRE_GRASP_MAX_WRIST_DELTA_DEG,
            PRE_GRASP_MAX_WRIST_DELTA_DEG,
        ))
        wr_delta = float(np.clip(
            raw_wr_delta * WRIST_ROLL_VISUAL_CORRECTION_SIGN * PRE_CLOSE_WRIST_DELTA_GAIN,
            -PRE_CLOSE_MAX_WRIST_STEP_DEG,
            PRE_CLOSE_MAX_WRIST_STEP_DEG,
        ))
        base_delta = float(np.clip(
            _safe_float(check.get("base_servo_delta_deg", 0.0), 0.0),
            -PRE_CLOSE_MAX_BASE_DELTA_DEG,
            PRE_CLOSE_MAX_BASE_DELTA_DEG,
        ))
        dy = float(np.clip(
            _safe_float(check.get("dy_mm", 0.0), 0.0),
            -PRE_CLOSE_MAX_Y_CORRECTION_MM,
            PRE_CLOSE_MAX_Y_CORRECTION_MM,
        ))
        entry = {
            "attempt": attempt,
            "aligned": bool(check.get("aligned")),
            "confidence": round(confidence, 2),
            "dx_mm": 0.0,
            "dy_mm": round(dy, 1),
            "base_servo_delta_deg": round(base_delta, 1),
            "wrist_roll_delta_deg": round(wr_delta, 1),
            "raw_image_wrist_roll_delta_deg": round(raw_wr_delta, 1),
            "wrist_delta_gain": PRE_CLOSE_WRIST_DELTA_GAIN,
            "reasoning": check.get("reasoning", ""),
        }
        history.append(entry)

        if bool(check.get("aligned")) and confidence >= PRE_GRASP_WRIST_CONFIDENCE_THRESHOLD:
            return {
                "success": True,
                "aligned": True,
                "target_x": round(current_x, 1),
                "target_y": round(current_y, 1),
                "target_z": round(current_z, 1),
                "wrist_roll_deg": round(current_wr, 1),
                "history": history,
            }

        if attempt >= max_corrections:
            break

        if abs(wr_delta) < 2.0 and abs(base_delta) < 1.0 and abs(dy) < 2.0:
            break

        if abs(dy) >= 2.0:
            current_y += dy
        elif abs(base_delta) >= 1.0:
            current_y, dy_from_base = y_target_from_base_delta(current_x, current_y, base_delta)
            entry["dy_from_base_delta_mm"] = round(dy_from_base, 1)

        current_wr = float(np.clip(
            current_wr + wr_delta,
            SERVO_CONFIG["wrist_roll"]["min"],
            SERVO_CONFIG["wrist_roll"]["max"],
        ))
        current_wr = clamp_wrist_roll_for_workspace(current_wr, current_x)
        move_arm_direct(current_x, current_y, current_z, current_wr)
        time.sleep(PRE_CLOSE_SETTLE_SECONDS)

    return {
        "success": True,
        "aligned": False,
        "target_x": round(current_x, 1),
        "target_y": round(current_y, 1),
        "target_z": round(current_z, 1),
        "wrist_roll_deg": round(current_wr, 1),
        "history": history,
        "note": "Use these returned target/wrist values for descent, or re-detect if confidence is poor.",
    }


def _is_side_grasp_object(text):
    t = str(text).lower()
    return any(token in t for token in (
        "bottle", "sticla", "sticlă", "test tube", "eprubeta", "eprubetă",
        "tube", "cup", "pahar", "cylinder", "cilindru", "can", "doza", "doză"
    ))


def _side_grasp_z_from_detection(obj, requested_z=None):
    if requested_z is not None:
        return float(np.clip(_safe_float(requested_z, SIDE_GRASP_DEFAULT_Z_MM),
                             SIDE_GRASP_MIN_Z_MM, SIDE_GRASP_MAX_Z_MM))
    h = _safe_float(obj.get("height_mm", 0.0), 0.0)
    if h <= 45.0 and _is_side_grasp_object(obj.get("label", "")):
        h = SIDE_GRASP_DEFAULT_Z_MM * 2.0
    if h <= 0.0:
        return SIDE_GRASP_DEFAULT_Z_MM
    return float(np.clip(h * 0.5, SIDE_GRASP_MIN_Z_MM, SIDE_GRASP_MAX_Z_MM))


def side_grasp_object_direct(object_description, grasp_z=None, lift_z=None):
    """Experimental side grasp primitive for bottles/test tubes/cylinders."""
    det = detect_objects_once(object_description, temperature=0.0)
    objects = det.get("objects", [])
    if not objects:
        return {"success": False, "error": "Object not detected", "detection": det}

    obj = objects[0]
    rx = _safe_float(obj.get("robot", {}).get("x"), 200.0)
    ry = _safe_float(obj.get("robot", {}).get("y"), 0.0)
    gz = _side_grasp_z_from_detection(obj, grasp_z)
    lift_z = float(np.clip(_safe_float(lift_z, SIDE_GRASP_LIFT_Z_MM), gz + 35.0, 220.0))
    pre_x = max(60.0, rx - SIDE_GRASP_PREAPPROACH_X_BACKOFF_MM)
    wp = SIDE_GRASP_WRIST_PITCH_DEG
    wr = SIDE_GRASP_WRIST_ROLL_DEG

    print(f"  Side grasp '{object_description}': target=({rx:.1f},{ry:.1f},{gz:.1f}) pre_x={pre_x:.1f} wp={wp:.1f} wr={wr:.1f}")
    set_gripper(True)
    time.sleep(0.4)
    move_arm_direct(pre_x, ry, SIDE_GRASP_TRANSIT_Z_MM, wr, wp)
    time.sleep(1.5)
    move_arm_direct(pre_x, ry, gz, wr, wp)
    time.sleep(1.5)
    move_arm_direct(rx, ry, gz, wr, wp)
    time.sleep(1.5)
    set_gripper(False)
    held_object_state.update({
        "holding": True,
        "grasp_mode": "side",
        "object_description": object_description,
        "wrist_pitch_deg": wp,
        "wrist_roll_deg": wr,
    })
    time.sleep(1.2)
    # Lift vertically first so the user can see the object is actually held.
    move_arm_direct(rx, ry, lift_z, wr, wp)
    time.sleep(1.8)
    move_arm_direct(pre_x, ry, lift_z, wr, wp)
    time.sleep(1.2)

    return {
        "success": True,
        "mode": "side_grasp",
        "holding_object": True,
        "lifted": True,
        "object": obj,
        "target": {"x": round(rx, 1), "y": round(ry, 1), "z": round(gz, 1)},
        "lift": {"x": round(rx, 1), "y": round(ry, 1), "z": round(lift_z, 1)},
        "preapproach": {"x": round(pre_x, 1), "y": round(ry, 1), "z": round(gz, 1)},
        "wrist_pitch_deg": wp,
        "wrist_roll_deg": wr,
        "note": "Experimental side grasp. Tune SIDE_GRASP_WRIST_PITCH_DEG if the gripper is not parallel to the table.",
    }


def point_grasp_y_offset_px(py, img_h):
    """Return a vertical grasp offset that changes across the camera workspace."""
    y_norm = float(py) / max(float(img_h), 1.0)
    if y_norm <= POINT_GRASP_TOP_Y_NORM:
        return POINT_GRASP_TOP_OFFSET_PX
    if y_norm >= POINT_GRASP_BOTTOM_Y_NORM:
        return POINT_GRASP_BOTTOM_OFFSET_PX
    if y_norm <= POINT_GRASP_MID_Y_NORM:
        t = (y_norm - POINT_GRASP_TOP_Y_NORM) / max(POINT_GRASP_MID_Y_NORM - POINT_GRASP_TOP_Y_NORM, 1e-6)
        return POINT_GRASP_TOP_OFFSET_PX + t * (POINT_GRASP_MID_OFFSET_PX - POINT_GRASP_TOP_OFFSET_PX)
    t = (y_norm - POINT_GRASP_MID_Y_NORM) / max(POINT_GRASP_BOTTOM_Y_NORM - POINT_GRASP_MID_Y_NORM, 1e-6)
    return POINT_GRASP_MID_OFFSET_PX + t * (POINT_GRASP_BOTTOM_OFFSET_PX - POINT_GRASP_MID_OFFSET_PX)


def point_grasp_x_offset_px(px, img_w):
    """Return a horizontal grasp offset relative to the camera center."""
    x_norm = float(px) / max(float(img_w), 1.0)
    if x_norm <= POINT_GRASP_LEFT_X_NORM:
        return POINT_GRASP_LEFT_OFFSET_PX
    if x_norm >= POINT_GRASP_RIGHT_X_NORM:
        return POINT_GRASP_RIGHT_OFFSET_PX
    if x_norm <= POINT_GRASP_CENTER_X_NORM:
        t = (x_norm - POINT_GRASP_LEFT_X_NORM) / max(POINT_GRASP_CENTER_X_NORM - POINT_GRASP_LEFT_X_NORM, 1e-6)
        return POINT_GRASP_LEFT_OFFSET_PX + t * (POINT_GRASP_CENTER_X_OFFSET_PX - POINT_GRASP_LEFT_OFFSET_PX)
    t = (x_norm - POINT_GRASP_CENTER_X_NORM) / max(POINT_GRASP_RIGHT_X_NORM - POINT_GRASP_CENTER_X_NORM, 1e-6)
    return POINT_GRASP_CENTER_X_OFFSET_PX + t * (POINT_GRASP_RIGHT_OFFSET_PX - POINT_GRASP_CENTER_X_OFFSET_PX)


def point_grasp_offset_px(px, py, img_w, img_h):
    return point_grasp_x_offset_px(px, img_w), point_grasp_y_offset_px(py, img_h)


def bbox_grasp_candidates(bbox_px, height_mm, img_w, img_h, grasp_fraction=0.5):
    """Return left/center/right candidate grasp points from a detector bbox."""
    if not bbox_px or bbox_px.get("w", 0) <= 0 or bbox_px.get("h", 0) <= 0:
        return []
    x0 = float(bbox_px["x"])
    y0 = float(bbox_px["y"])
    w = float(bbox_px["w"])
    h = float(bbox_px["h"])
    y = y0 + h * float(grasp_fraction)
    candidates = []
    for name, frac in (("left_inset", 0.18), ("center", 0.50), ("right_inset", 0.82)):
        x = x0 + w * frac
        x = float(np.clip(x, 0, max(0, img_w - 1)))
        y_clamped = float(np.clip(y, 0, max(0, img_h - 1)))
        robot_xy = pixel_to_robot(x, y_clamped, height_mm)
        if not robot_xy:
            continue
        rx, ry = robot_xy
        candidates.append({
            "name": name,
            "bbox_fraction_x": frac,
            "pixel": {"x": int(round(x)), "y": int(round(y_clamped))},
            "robot": {"x": round(rx, 1), "y": round(ry, 1), "z": round(float(height_mm), 1)},
        })
    return candidates


def normalize_angle_180(deg):
    return ((float(deg) + 180.0) % 360.0) - 180.0


def normalize_axis_angle_90(deg):
    """Normalize an unoriented object axis angle to [-90, 90)."""
    angle = normalize_angle_180(deg)
    while angle < -90.0:
        angle += 180.0
    while angle >= 90.0:
        angle -= 180.0
    return angle


def contour_axis_angle_deg(contour):
    """Estimate the main visual axis angle of a contour in image pixels."""
    try:
        rect = cv2.minAreaRect(contour)
        (rw, rh) = rect[1]
        if rw < 8 or rh < 8:
            return None
        angle = float(rect[2])
        if rw < rh:
            angle += 90.0
        return normalize_axis_angle_90(angle)
    except Exception:
        return None


def image_axis_to_robot_table_angle(px, py, image_axis_deg, z_robot_mm=0.0):
    """Convert an image-space object axis into a robot-table X/Y angle."""
    if image_axis_deg is None:
        return None
    length_px = 45.0
    theta = np.radians(float(image_axis_deg))
    dx = np.cos(theta) * length_px
    dy = np.sin(theta) * length_px
    p1 = pixel_to_robot(float(px) - dx, float(py) - dy, z_robot_mm)
    p2 = pixel_to_robot(float(px) + dx, float(py) + dy, z_robot_mm)
    if not p1 or not p2:
        return None
    rx1, ry1 = p1
    rx2, ry2 = p2
    if abs(rx2 - rx1) < 1e-6 and abs(ry2 - ry1) < 1e-6:
        return None
    return normalize_axis_angle_90(np.degrees(np.arctan2(ry2 - ry1, rx2 - rx1)))


def plan_top_grasp_wrist_roll(rx, ry, table_orientation_deg=None):
    """Plan a modest top-grasp wrist roll from the object's table angle."""
    wr_cfg = SERVO_CONFIG["wrist_roll"]
    rx = float(rx)
    ry = float(ry)
    arm_bearing = normalize_angle_180(np.degrees(np.arctan2(float(ry), max(float(rx), 1e-6))))
    if table_orientation_deg is None:
        table_orientation_deg = 0.0
        source = "neutral_no_orientation"
    else:
        table_orientation_deg = normalize_axis_angle_90(table_orientation_deg)
        source = "table_orientation"
    relative_axis = normalize_axis_angle_90(table_orientation_deg)
    workspace_scale = 1.0
    max_delta = WRIST_ROLL_MAX_PLANNER_DELTA_DEG
    workspace_zone = "center"
    if rx < WRIST_ROLL_EDGE_X_LOW_MM or rx > WRIST_ROLL_EDGE_X_HIGH_MM:
        max_delta = WRIST_ROLL_EDGE_MAX_PLANNER_DELTA_DEG
        workspace_zone = "edge_x"
    planner_delta = float(np.clip(
        WRIST_ROLL_OBJECT_TO_SERVO_SIGN * WRIST_ROLL_ALIGNMENT_GAIN * workspace_scale * relative_axis,
        -max_delta,
        max_delta,
    ))
    wrist_roll = WRIST_ROLL_NEUTRAL_DEG + WRIST_ROLL_CALIBRATION_BIAS_DEG + planner_delta
    wrist_roll = float(np.clip(wrist_roll, wr_cfg["min"], wr_cfg["max"]))
    return {
        "grasp_mode": "top",
        "wrist_roll_deg": round(wrist_roll, 1),
        "neutral_deg": WRIST_ROLL_NEUTRAL_DEG,
        "planner_delta_deg": round(planner_delta, 1),
        "max_delta_deg": round(float(max_delta), 1),
        "workspace_zone": workspace_zone,
        "workspace_scale": round(workspace_scale, 2),
        "arm_bearing_deg": round(arm_bearing, 1),
        "table_orientation_deg": round(float(table_orientation_deg), 1),
        "source": source,
    }


def clamp_wrist_roll_for_workspace(wrist_roll_deg, rx):
    """Keep edge-workspace wrist roll away from extreme values caused by perspective."""
    wrist_roll = float(wrist_roll_deg)
    rx = float(rx)
    if rx < WRIST_ROLL_EDGE_X_LOW_MM or rx > WRIST_ROLL_EDGE_X_HIGH_MM:
        lo = WRIST_ROLL_NEUTRAL_DEG - WRIST_ROLL_EDGE_MAX_FINAL_DELTA_DEG
        hi = WRIST_ROLL_NEUTRAL_DEG + WRIST_ROLL_EDGE_MAX_FINAL_DELTA_DEG
        return float(np.clip(wrist_roll, lo, hi))
    return wrist_roll


def refine_object_pixel_with_color(frame, px, py, label="", query="", bbox_px=None):
    """Refine a rough VLM detection to the local object contour center."""
    text = f"{label} {query}".lower()
    is_cube_like = any(token in text for token in ("cube", "cub", "block", "rubik", "3x3", "3 x 3", "2x2", "2 x 2"))

    h, w = frame.shape[:2]
    if bbox_px and bbox_px.get("w", 0) > 8 and bbox_px.get("h", 0) > 8:
        pad = int(max(45, 0.45 * max(bbox_px["w"], bbox_px["h"])))
        x1 = max(0, int(bbox_px["x"] - pad))
        y1 = max(0, int(bbox_px["y"] - pad))
        x2 = min(w, int(bbox_px["x"] + bbox_px["w"] + pad))
        y2 = min(h, int(bbox_px["y"] + bbox_px["h"] + pad))
    else:
        # Point-only detections should refine only near Gemini's point. A large
        # radius can jump to a neighboring cube and make the robot chase ghosts.
        radius = 135
        x1 = max(0, int(px - radius))
        y1 = max(0, int(py - radius))
        x2 = min(w, int(px + radius))
        y2 = min(h, int(py + radius))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # The table is bright/low saturation. Score masks separately so an orange
    # debug overlay or one noisy edge cannot merge the whole ROI into one blob.
    color_mask = ((hsv[:, :, 1] > 42) & (hsv[:, :, 2] > 35))
    dark_mask = (hsv[:, :, 2] < 150)
    edges = cv2.Canny(gray, 45, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    mask_sources = [
        ("color", color_mask.astype(np.uint8) * 255),
        ("dark", dark_mask.astype(np.uint8) * 255),
        ("edge", edges),
    ]
    kernel = np.ones((5, 5), np.uint8)
    candidates = []
    best = None
    best_score = -1.0
    local_px = px - x1
    local_py = py - y1
    roi_area = max((x2 - x1) * (y2 - y1), 1)
    local_bbox = None
    if bbox_px:
        local_bbox = (
            max(0, bbox_px["x"] - x1),
            max(0, bbox_px["y"] - y1),
            min(x2 - x1, bbox_px["x"] + bbox_px["w"] - x1),
            min(y2 - y1, bbox_px["y"] + bbox_px["h"] - y1),
        )

    for mask_name, mask in mask_sources:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 450:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < 18 or bh < 18:
                continue
            if (bw * bh) > roi_area * 0.72:
                continue
            aspect = bw / max(bh, 1)
            if aspect < 0.25 or aspect > 4.8:
                continue
            extent = area / max(float(bw * bh), 1.0)
            if extent < 0.10:
                continue
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            dist = float(((cx - local_px) ** 2 + (cy - local_py) ** 2) ** 0.5)
            dy_from_click = cy - local_py
            compact_bonus = min(aspect, 1.0 / aspect)
            score = area * (0.55 + compact_bonus) * (0.65 + extent) / (1.0 + (dist / 85.0) ** 2)
            overlap_ratio = None
            if local_bbox is not None:
                lx1, ly1, lx2, ly2 = local_bbox
                ix1 = max(bx, lx1)
                iy1 = max(by, ly1)
                ix2 = min(bx + bw, lx2)
                iy2 = min(by + bh, ly2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                overlap_ratio = inter / max(float(bw * bh), 1.0)
                score *= max(0.12, min(1.0, overlap_ratio * 1.45))
            if mask_name == "edge":
                score *= 0.55
            elif mask_name == "dark":
                score *= 1.15
            # If Gemini lands below/front of the object, prefer the object blob just above it.
            if dy_from_click < -12:
                score *= 1.45 if is_cube_like else 1.2
            elif dy_from_click <= 90:
                score *= 1.12
            elif dy_from_click > 135:
                score *= 0.45
            candidates.append({
                "x": int(x1 + bx),
                "y": int(y1 + by),
                "w": int(bw),
                "h": int(bh),
                "area": round(area, 1),
                "aspect": round(aspect, 2),
                "extent": round(extent, 2),
                "score": round(score, 1),
                "mask": mask_name,
                "bbox_overlap": round(overlap_ratio, 2) if overlap_ratio is not None else None,
                "dy_from_click": round(float(dy_from_click), 1),
            })
            if score > best_score:
                best_score = score
                best = (bx, by, bw, bh, area, contour)

    if best is None:
        return None

    bx, by, bw, bh, area, best_contour = best
    center_x = x1 + bx + bw / 2.0
    center_y = y1 + by + bh / 2.0
    grasp_fraction = CUBE_BBOX_GRASP_Y_FRACTION if is_cube_like else OBJECT_GRASP_Y_FRACTION
    grasp_x = center_x
    grasp_y = y1 + by + bh * grasp_fraction
    orientation_image_deg = contour_axis_angle_deg(best_contour)
    return {
        "px": int(round(grasp_x)),
        "py": int(round(grasp_y)),
        "center_px": {"x": int(round(center_x)), "y": int(round(center_y))},
        "grasp_px": {"x": int(round(grasp_x)), "y": int(round(grasp_y))},
        "grasp_y_fraction": grasp_fraction,
        "source": "local_contour_grasp",
        "visual_target_mode": "object_center",
        "area_px": round(area, 1),
        "orientation_image_deg": round(float(orientation_image_deg), 1) if orientation_image_deg is not None else None,
        "candidate_count": len(candidates),
        "candidates": sorted(candidates, key=lambda c: c["score"], reverse=True)[:5],
        "bbox_px": {
            "x": int(x1 + bx),
            "y": int(y1 + by),
            "w": int(bw),
            "h": int(bh),
        },
    }


def detect_objects_once(query, temperature=0.0):
    import urllib.request as urlreq

    frame = get_camera_frame()
    if frame is None:
        return {"success": False, "error": "Camera not running", "objects": []}
    img_h, img_w = frame.shape[:2]
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

    prompt = f"""Find the center of: {query}
Return ONLY valid JSON, no markdown.

Return the center point of each relevant physical object.
Do not return a point on the table, edge, side face, shadow, label, highlight, or internal grid.
For cubes, return the center of the whole cube, not one colored cell.
For bottles, test tubes, cups, and cylinders, return a point near the object's visual center and estimate the real object height_mm.
Also estimate orientation_deg: the angle of the object's main top-face edge on the table.
Do not copy 0 unless the object is actually straight. For a visibly rotated cube, return a non-zero orientation.

Use this exact JSON format:
[{{"point":[y,x],"label":"object name","height_mm":35,"orientation_deg":15}}]

Coordinates are normalized 0-1000 in [y,x] order.
The point must be on the object itself. Do not compensate for robot mechanics."""

    request_body = {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
            {"text": prompt}
        ]}],
        "generationConfig": {"temperature": temperature, "thinkingConfig": {"thinkingBudget": 0}}
    }
    det_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    req = urlreq.Request(det_url, data=json.dumps(request_body).encode('utf-8'),
                         headers={"Content-Type": "application/json"}, method="POST")
    with urlreq.urlopen(req, timeout=15) as resp:
        det_result = json.loads(resp.read().decode('utf-8'))

    text = ""
    for candidate in det_result.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                text += part["text"]

    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()

    try:
        raw_points = json.loads(clean)
    except Exception:
        return {"success": True, "objects": [], "raw_response": text}

    objects = []
    debug_frame = frame.copy()
    for p in raw_points:
        center_source = "point"
        label_text = str(p.get("label", ""))
        cube_like = any(token in f"{label_text} {query}".lower()
                        for token in ("cube", "cub", "block", "rubik", "3x3", "3 x 3", "2x2", "2 x 2"))
        if "quad" in p and isinstance(p["quad"], list) and len(p["quad"]) >= 4:
            try:
                quad = np.array(p["quad"], dtype=np.float64)
                ny = float(np.mean(quad[:, 0]))
                nx = float(np.mean(quad[:, 1]))
                center_source = "quad_centroid"
            except Exception:
                continue
        else:
            bbox = None
            for bbox_key in ("box_2d", "bbox", "bounding_box", "box"):
                if bbox_key in p and isinstance(p[bbox_key], list) and len(p[bbox_key]) >= 4:
                    bbox = p[bbox_key]
                    break
            if bbox is not None:
                try:
                    y_min, x_min, y_max, x_max = [float(v) for v in bbox[:4]]
                    y_min, y_max = sorted((max(0.0, min(1000.0, y_min)), max(0.0, min(1000.0, y_max))))
                    x_min, x_max = sorted((max(0.0, min(1000.0, x_min)), max(0.0, min(1000.0, x_max))))
                    nx = (x_min + x_max) / 2.0
                    if cube_like:
                        ny = y_min + (y_max - y_min) * CUBE_BBOX_GRASP_Y_FRACTION
                        center_source = f"{bbox_key}_cube_grasp"
                    else:
                        ny = y_min + (y_max - y_min) * OBJECT_GRASP_Y_FRACTION
                        center_source = f"{bbox_key}_grasp"
                except Exception:
                    continue
            elif "center" in p:
                ny, nx = p["center"]
                center_source = "center"
            elif "point" in p:
                ny, nx = p["point"]
                center_source = "point"
            else:
                continue

        # Legacy branch kept out of the main path: Gemini boxes are the preferred center source.
        if False and "bbox" in p and isinstance(p["bbox"], list) and len(p["bbox"]) >= 4:
            try:
                y_min, x_min, y_max, x_max = [float(v) for v in p["bbox"][:4]]
                ny = (y_min + y_max) / 2.0
                nx = (x_min + x_max) / 2.0
                center_source = "bbox_center"
            except Exception:
                continue

        ny = max(0.0, min(1000.0, float(ny)))
        nx = max(0.0, min(1000.0, float(nx)))
        px = int(float(nx) * img_w / 1000)
        py = int(float(ny) * img_h / 1000)
        raw_px, raw_py = px, py
        bbox_px = None
        bbox_norm = None
        if "box_2d" in p or "bbox" in p or "bounding_box" in p or "box" in p:
            box_data = p.get("box_2d") or p.get("bbox") or p.get("bounding_box") or p.get("box")
            if isinstance(box_data, list) and len(box_data) >= 4:
                by1, bx1, by2, bx2 = [float(v) for v in box_data[:4]]
                bx1 = int(max(0, min(img_w - 1, bx1 * img_w / 1000)))
                bx2 = int(max(0, min(img_w - 1, bx2 * img_w / 1000)))
                by1 = int(max(0, min(img_h - 1, by1 * img_h / 1000)))
                by2 = int(max(0, min(img_h - 1, by2 * img_h / 1000)))
                x_a, x_b = sorted((bx1, bx2))
                y_a, y_b = sorted((by1, by2))
                bbox_px = {"x": x_a, "y": y_a, "w": x_b - x_a, "h": y_b - y_a}
                bbox_norm = {
                    "x": round(x_a / img_w * 100, 2),
                    "y": round(y_a / img_h * 100, 2),
                    "w": round((x_b - x_a) / img_w * 100, 2),
                    "h": round((y_b - y_a) / img_h * 100, 2),
                }

        refine = refine_object_pixel_with_color(frame, px, py, p.get("label", ""), query, bbox_px)
        refinement_failed = None
        if refine is not None:
            px, py = refine["px"], refine["py"]
            nx = px * 1000.0 / img_w
            ny = py * 1000.0 / img_h
            center_source = refine["source"]
        else:
            if bbox_px is None:
                x_offset_px, y_offset_px = 0.0, 0.0
                nx = px * 1000.0 / img_w
                ny = py * 1000.0 / img_h
                center_source = f"{center_source}_visual_center"
                refinement_failed = {
                    "reason": "point_only_no_local_contour_using_raw_object_point",
                    "dynamic_x_offset_px": round(float(x_offset_px), 1),
                    "dynamic_y_offset_px": round(float(y_offset_px), 1),
                }
            else:
                refinement_failed = "bbox_no_local_contour"
        height_mm = p.get("height_mm", 35)
        try:
            height_mm = float(height_mm)
        except (TypeError, ValueError):
            height_mm = 35.0
        height_limit = 180.0 if _is_side_grasp_object(f"{p.get('label', '')} {query}") else 80.0
        height_mm = max(0.0, min(height_limit, height_mm))
        robot_xy = pixel_to_robot(px, py, height_mm)
        if not robot_xy:
            continue
        rx, ry = robot_xy
        grasp_px_before_bias = {"x": int(px), "y": int(py)}
        robot_bias = {"x": 0.0, "y": 0.0}
        x_bias_mm = CUBE_GRASP_X_BIAS_MM if cube_like else OBJECT_GRASP_X_BIAS_MM
        if abs(x_bias_mm) > 0.001:
            rx += x_bias_mm
            robot_bias["x"] = x_bias_mm
            # Keep the displayed detection point on the visual grasp point.
            # Robot-space X compensation is real, but reprojecting it made the
            # Live overlay look like the detector had clicked on the table.
        bbox_candidates = bbox_grasp_candidates(
            bbox_px,
            height_mm,
            img_w,
            img_h,
            CUBE_BBOX_GRASP_Y_FRACTION if cube_like else OBJECT_GRASP_Y_FRACTION,
        )
        raw_ori = p.get("orientation_deg", 0)
        try:
            raw_ori = float(raw_ori)
        except (TypeError, ValueError):
            raw_ori = 0.0
        if abs(raw_ori) < 8:
            raw_ori = 0.0
        table_orientation = None
        orientation_source = "none"
        if refine and refine.get("orientation_image_deg") is not None:
            table_orientation = image_axis_to_robot_table_angle(px, py, refine["orientation_image_deg"], height_mm)
            if table_orientation is not None:
                orientation_source = "local_contour_robot_table"
        if table_orientation is None and raw_ori != 0.0:
            table_orientation = raw_ori
            orientation_source = "gemini_fallback_assumed_table"
        wrist_plan = plan_top_grasp_wrist_roll(rx, ry, table_orientation)
        grasp_mode = "side" if _is_side_grasp_object(f"{p.get('label', '')} {query}") else wrist_plan["grasp_mode"]
        if table_orientation is None:
            table_orientation = wrist_plan["table_orientation_deg"]
        if bbox_px is not None:
            cv2.rectangle(debug_frame,
                          (bbox_px["x"], bbox_px["y"]),
                          (bbox_px["x"] + bbox_px["w"], bbox_px["y"] + bbox_px["h"]),
                          (0, 140, 255), 2)
            for cand in bbox_candidates:
                cp = cand["pixel"]
                cv2.circle(debug_frame, (cp["x"], cp["y"]), 5, (0, 220, 255), -1)
        cv2.circle(debug_frame, (int(raw_px), int(raw_py)), 7, (255, 0, 255), -1)
        if refine and refine.get("center_px"):
            cv2.circle(debug_frame, (refine["center_px"]["x"], refine["center_px"]["y"]), 7, (255, 180, 0), -1)
        if refine and refine.get("orientation_image_deg") is not None:
            theta = np.radians(float(refine["orientation_image_deg"]))
            line_len = 45
            lx = int(np.cos(theta) * line_len)
            ly = int(np.sin(theta) * line_len)
            cv2.line(debug_frame, (int(px) - lx, int(py) - ly), (int(px) + lx, int(py) + ly), (0, 255, 255), 2)
        cv2.circle(debug_frame, (grasp_px_before_bias["x"], grasp_px_before_bias["y"]), 8, (255, 80, 0), -1)
        cv2.circle(debug_frame, (int(px), int(py)), 10, (0, 120, 255), -1)
        cv2.putText(debug_frame, f"{p.get('label', 'object')} {center_source} wr={wrist_plan['wrist_roll_deg']}",
                    (int(px) + 12, max(20, int(py) - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
        objects.append({
            "label": p.get("label", "unknown"),
            "pixel": {"x": px, "y": py},
            "raw_pixel": {"x": raw_px, "y": raw_py},
            "visual_center_pixel": refine.get("center_px") if refine else None,
            "grasp_pixel_before_bias": grasp_px_before_bias,
            "norm": {"x": round(float(nx) / 10.0, 2), "y": round(float(ny) / 10.0, 2)},
            "center_source": center_source,
            "visual_target_mode": "object_center_no_hidden_offset",
            "cube_like": cube_like,
            "robot_bias_mm": robot_bias,
            "bbox_px": bbox_px,
            "bbox_norm": bbox_norm,
            "bbox_grasp_candidates": bbox_candidates,
            "refinement": refine,
            "refinement_failed": refinement_failed,
            "robot": {"x": round(rx, 1), "y": round(ry, 1), "z": round(height_mm, 1)},
            "height_mm": round(height_mm, 1),
            "orientation_deg": round(float(table_orientation), 1),
            "raw_orientation_deg": round(raw_ori, 1),
            "orientation_source": orientation_source,
            "table_orientation_deg": round(float(table_orientation), 1),
            "wrist_roll_suggestion_deg": wrist_plan["wrist_roll_deg"],
            "wrist_roll_plan": wrist_plan,
            "grasp_mode": grasp_mode,
        })
    if objects:
        cv2.imwrite(VISION_DETECTION_DEBUG_PATH, debug_frame)
    return {
        "success": True,
        "objects": objects,
        "count": len(objects),
        "raw_response": text,
        "debug_image": VISION_DETECTION_DEBUG_PATH if objects else None,
    }


def stable_detect_object(query, samples=3):
    def choose_object(objects):
        if not objects:
            return None
        query_tokens = [t for t in str(query).lower().replace("-", " ").split() if len(t) >= 3]
        if not query_tokens:
            return objects[0]
        best = objects[0]
        best_score = -1
        for obj in objects:
            label = str(obj.get("label", "")).lower()
            score = sum(1 for token in query_tokens if token in label)
            if score > best_score:
                best = obj
                best_score = score
        return best

    detections = []
    raw_counts = []
    for i in range(max(1, samples)):
        result = detect_objects_once(query, temperature=0.0)
        objects = result.get("objects", [])
        raw_counts.append(len(objects))
        if objects:
            detections.append(choose_object(objects))
        time.sleep(0.25)

    if not detections:
        return {"success": False, "error": "Object not detected", "objects_seen": raw_counts}

    xs = np.array([d["robot"]["x"] for d in detections], dtype=np.float64)
    ys = np.array([d["robot"]["y"] for d in detections], dtype=np.float64)
    hs = np.array([d.get("height_mm", 35) for d in detections], dtype=np.float64)
    oris = np.array([d.get("orientation_deg", 0) for d in detections], dtype=np.float64)
    wrists = np.array([d.get("wrist_roll_suggestion_deg", WRIST_ROLL_NEUTRAL_DEG) for d in detections], dtype=np.float64)
    norm_xs = np.array([d.get("norm", {}).get("x", 0) for d in detections], dtype=np.float64)
    norm_ys = np.array([d.get("norm", {}).get("y", 0) for d in detections], dtype=np.float64)

    med_xy = np.array([np.median(xs), np.median(ys)])
    dists = np.sqrt((xs - med_xy[0]) ** 2 + (ys - med_xy[1]) ** 2)
    keep = dists <= 35.0
    if np.sum(keep) >= 2:
        xs, ys, hs, oris, wrists = xs[keep], ys[keep], hs[keep], oris[keep], wrists[keep]
        norm_xs, norm_ys = norm_xs[keep], norm_ys[keep]
        kept = int(np.sum(keep))
    else:
        kept = len(detections)

    obj = {
        "label": detections[0].get("label", query),
        "robot": {
            "x": round(float(np.median(xs)), 1),
            "y": round(float(np.median(ys)), 1),
            "z": round(float(np.median(hs)), 1),
        },
        "height_mm": round(float(np.median(hs)), 1),
        "orientation_deg": round(float(np.median(oris)), 1),
        "table_orientation_deg": round(float(np.median(oris)), 1),
        "wrist_roll_suggestion_deg": round(float(np.median(wrists)), 1),
        "grasp_mode": detections[0].get("grasp_mode", "top"),
        "bbox_grasp_candidates": detections[0].get("bbox_grasp_candidates", []),
        "bbox_px": detections[0].get("bbox_px"),
        "norm": {"x": round(float(np.median(norm_xs)), 2), "y": round(float(np.median(norm_ys)), 2)},
        "detections": detections,
        "samples": len(detections),
        "kept_samples": kept,
        "spread_mm": round(float(np.max(dists)) if len(dists) else 0.0, 1),
    }
    return {"success": True, "object": obj}


def agent_execute_function(fn_name, fn_args):
    """Execute a function called by the agent. Returns result dict."""
    import urllib.request as urlreq

    try:
        if fn_name == "move_arm":
            x, y, z = fn_args.get("x", 200), fn_args.get("y", 0), fn_args.get("z", 200)
            wrist_roll_deg = fn_args.get("wrist_roll_deg", None)
            wrist_pitch_deg = fn_args.get("wrist_pitch_deg", None)
            return move_arm_direct(x, y, z, wrist_roll_deg, wrist_pitch_deg)

        elif fn_name == "open_gripper":
            return set_gripper(True)

        elif fn_name == "close_gripper":
            return set_gripper(False)

        elif fn_name == "wave_hello":
            run_hello_wave()
            return {"success": True, "message": "IRIS greeted the audience."}

        elif fn_name == "move_to_safe_height":
            return move_arm_direct(200, 0, 200)

        elif fn_name == "go_home":
            send_raw(0, 0)
            with move_lock:
                current_angles[0] = 0
                target_angles[0] = 0
            time.sleep(0.3)
            move_arm_direct(200, 0, 250)
            return {"success": True}

        elif fn_name == "wait_seconds":
            seconds = max(0.1, min(10, fn_args.get("seconds", 1)))
            time.sleep(seconds)
            return {"success": True, "waited": seconds}

        elif fn_name == "detect_objects":
            query = fn_args.get("query", "all objects")
            samples = int(max(1, min(5, fn_args.get("samples", 1))))
            if samples > 1:
                stable = stable_detect_object(query, samples=samples)
                if stable.get("success"):
                    stable["objects"] = [stable["object"]]
                    stable["count"] = 1
                return stable
            return detect_objects_once(query, temperature=0.0)

        elif fn_name == "check_success":
            task = fn_args.get("task_description", "")
            frame = get_camera_frame()
            if frame is None:
                return {"success": False, "error": "Camera not running"}
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

            prompt = f"""Look at this image from a robot's camera.
Determine if the following task has been completed successfully: "{task}"
Respond in JSON format: {{"completed": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

            request_body = {
                "contents": [{"parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                    {"text": prompt}
                ]}],
                "generationConfig": {"temperature": 0.3, "thinkingConfig": {"thinkingBudget": 1024}}
            }
            chk_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
            req = urlreq.Request(chk_url, data=json.dumps(request_body).encode('utf-8'),
                                 headers={"Content-Type": "application/json"}, method="POST")
            with urlreq.urlopen(req, timeout=15) as resp:
                chk_result = json.loads(resp.read().decode('utf-8'))

            text = ""
            for candidate in chk_result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part:
                        text += part["text"]
            clean = text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                clean = clean.rsplit("```", 1)[0]
            try:
                check = json.loads(clean.strip())
                return {"success": True, **check}
            except:
                return {"success": True, "completed": False, "confidence": 0, "reasoning": text[:200]}

        elif fn_name == "get_gripper_pose":
            return get_visual_gripper_pose()

        elif fn_name == "check_target_alignment":
            target_x = fn_args.get("target_x")
            target_y = fn_args.get("target_y")
            if target_x is None or target_y is None:
                return {"success": False, "error": "target_x and target_y are required"}
            target_z = fn_args.get("target_z", 0.0)
            tolerance = fn_args.get("tolerance_mm", 5.0)
            return check_target_alignment(target_x, target_y, target_z, tolerance)

        elif fn_name == "side_grasp_object":
            return side_grasp_object_direct(
                fn_args.get("object_description", "object"),
                fn_args.get("grasp_z", None),
                fn_args.get("lift_z", None),
            )

        elif fn_name == "describe_scene":
            frame = get_camera_frame()
            if frame is None:
                return {"success": False, "error": "Camera not running"}
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

            prompt = """Describe what you see on this table/workspace from a robot arm's camera.
List all visible objects with approximate positions. Be concise. Respond in Romanian."""

            request_body = {
                "contents": [{"parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                    {"text": prompt}
                ]}],
                "generationConfig": {"temperature": 0.5, "thinkingConfig": {"thinkingBudget": 512}}
            }
            desc_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
            req = urlreq.Request(desc_url, data=json.dumps(request_body).encode('utf-8'),
                                 headers={"Content-Type": "application/json"}, method="POST")
            with urlreq.urlopen(req, timeout=15) as resp:
                desc_result = json.loads(resp.read().decode('utf-8'))

            text = ""
            for candidate in desc_result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part:
                        text += part["text"]
            return {"success": True, "description": text}

        elif fn_name == "set_servo_direct":
            channel = fn_args.get("channel", 6)
            angle = fn_args.get("angle", 90)
            # Safety clamp per channel
            limits = {6: (0, 180), 4: (0, 160), 3: (20, 180), 2: (0, 170), 1: (0, 180), 0: (0, 70)}
            lo, hi = limits.get(channel, (0, 180))
            angle = max(lo, min(hi, angle))
            set_target(channel, angle)
            print(f"  Agent → Servo CH{channel} = {angle}°")
            return {"success": True, "channel": channel, "angle": angle}

        elif fn_name == "trim_base_rotation":
            delta = float(fn_args.get("delta_deg", 0))
            # Cap delta so the agent can't swing the base.
            delta = max(-FINAL_ALIGN_MAX_BASE_DELTA_DEG, min(FINAL_ALIGN_MAX_BASE_DELTA_DEG, delta))
            cfg = SERVO_CONFIG['base_rotation']
            current = target_angles.get(6, current_angles.get(6, cfg['offset']))
            new_angle = max(cfg['min'], min(cfg['max'], current + delta))
            set_target(6, new_angle)
            print(f"  Agent → Base trim Δ={delta:+.1f}° ({current:.1f}° → {new_angle:.1f}°)")
            return {"success": True, "delta_deg": delta, "previous_angle": round(current, 1), "new_angle": round(new_angle, 1)}

        elif fn_name == "check_gripper_proximity":
            obj_desc = fn_args.get("object_description", "object")
            frame = get_camera_frame()
            if frame is None:
                return {"success": False, "error": "Camera not running"}
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

            prompt = f"""Look at this image from a robot arm camera. The gripper is currently lowered near an object.
Determine: Is the "{obj_desc}" positioned correctly between the gripper jaws so that closing the gripper would successfully grab it?
Use the robot coordinate system:
- X larger = farther forward from the robot base.
- X smaller = back toward the robot base.
- Y larger = left.
- Y smaller = right.

This is a final sanity check only. Do not ask for motion unless the object is clearly outside the gripper.
DON'T OVERTHINK IT. If the object is visibly between the gripper jaws, mark aligned=true even if it is not visually perfect.
Do not suggest tiny cosmetic corrections. If the gripper can probably close on it, accept it.
Corrections are handled earlier by align_before_descent. Here, prefer aligned=true or requires redetection.
If rotation is badly wrong, mention it in reasoning but keep wrist_roll_delta_deg=0 here.
For wrist_roll_delta_deg, use image-space signs:
- positive = gripper should rotate clockwise in the camera image.
- negative = gripper should rotate counter-clockwise in the camera image.
Respond ONLY in JSON:
{{
  "aligned": true/false,
  "confidence": 0.0-1.0,
  "dx_mm": 0,
  "dy_mm": 0,
  "wrist_roll_delta_deg": 0,
  "base_servo_delta_deg": 0,
  "reasoning": "brief Romanian explanation"
}}"""

            request_body = {
                "contents": [{"parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                    {"text": prompt}
                ]}],
                "generationConfig": {"temperature": 0.3, "thinkingConfig": {"thinkingBudget": 1024}}
            }
            prox_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
            req = urlreq.Request(prox_url, data=json.dumps(request_body).encode('utf-8'),
                                 headers={"Content-Type": "application/json"}, method="POST")
            with urlreq.urlopen(req, timeout=15) as resp:
                prox_result = json.loads(resp.read().decode('utf-8'))

            text = ""
            for candidate in prox_result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part:
                        text += part["text"]
            clean = text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                clean = clean.rsplit("```", 1)[0]
            try:
                check = json.loads(clean.strip())
                raw_wr_delta = float(np.clip(
                    _safe_float(check.get("wrist_roll_delta_deg", 0.0), 0.0),
                    -FINAL_ALIGN_MAX_WRIST_DELTA_DEG,
                    FINAL_ALIGN_MAX_WRIST_DELTA_DEG,
                ))
                check["raw_image_wrist_roll_delta_deg"] = round(raw_wr_delta, 1)
                check["wrist_roll_delta_deg"] = round(float(np.clip(
                    raw_wr_delta * WRIST_ROLL_VISUAL_CORRECTION_SIGN,
                    -FINAL_ALIGN_MAX_WRIST_DELTA_DEG,
                    FINAL_ALIGN_MAX_WRIST_DELTA_DEG,
                )), 1)
                check["visual_correction_sign"] = WRIST_ROLL_VISUAL_CORRECTION_SIGN
                return {"success": True, **check}
            except:
                return {"success": True, "aligned": False, "confidence": 0, "adjustment": text[:200]}

        elif fn_name == "check_wrist_roll_alignment":
            return check_wrist_roll_alignment_direct(
                fn_args.get("object_description", "object"),
                fn_args.get("current_wrist_roll_deg", last_wrist_roll[0]),
            )

        elif fn_name == "align_before_descent":
            return align_before_descent_direct(
                fn_args.get("object_description", "object"),
                fn_args.get("target_x", 200),
                fn_args.get("target_y", 0),
                fn_args.get("current_wrist_roll_deg", last_wrist_roll[0]),
                fn_args.get("target_z", PRE_GRASP_ROTATE_Z_MM),
            )

        elif fn_name == "align_gripper_for_close":
            return align_gripper_for_close_direct(
                fn_args.get("object_description", "object"),
                fn_args.get("target_x", 200),
                fn_args.get("target_y", 0),
                fn_args.get("target_z", 0),
                fn_args.get("wrist_roll_deg", None),
                fn_args.get("max_corrections", FINAL_ALIGN_MAX_CORRECTIONS),
            )

        else:
            return {"error": f"Unknown function: {fn_name}"}

    except Exception as e:
        print(f"  Agent function error ({fn_name}): {e}")
        return {"success": False, "error": str(e)}

def ik_to_servo(ik_angles_rad):
    """Convert IK radians to servo degrees dict {ch: angle}"""
    result = {}
    for i, name in enumerate(JOINT_NAMES):
        cfg = SERVO_CONFIG[name]
        ik_deg = np.degrees(ik_angles_rad[i + 1])
        servo_deg = cfg['offset'] + cfg['dir'] * ik_deg
        servo_deg = int(round(np.clip(servo_deg, cfg['min'], cfg['max'])))
        result[cfg['ch']] = servo_deg
    # CH5 mirror
    result[5] = 180 - result[4]
    return result

def compensate_z(target_xyz):
    """Compensate Z based on calibration: IK Z != real Z.
    Measured calibration points (gripper touching table = Z_real ≈ 0):
      X=90mm  → need Z_ik=30mm to reach table
      X=200mm → need Z_ik=14mm to reach table
      X=350mm → need Z_ik=-10mm to reach table
    Returns adjusted [x, y, z] in meters."""
    x_mm = target_xyz[0] * 1000
    z_mm = target_xyz[2] * 1000

    # Linear interpolation of Z offset based on X
    # Calibration points: (X_mm, Z_offset_mm to add)
    cal = [(90, 30), (200, 14), (350, -10)]

    if x_mm <= cal[0][0]:
        offset = cal[0][1]
    elif x_mm >= cal[-1][0]:
        offset = cal[-1][1]
    else:
        # Find bracketing points and interpolate
        for i in range(len(cal) - 1):
            x0, o0 = cal[i]
            x1, o1 = cal[i + 1]
            if x0 <= x_mm <= x1:
                t = (x_mm - x0) / (x1 - x0)
                offset = o0 + t * (o1 - o0)
                break

    z_compensated = z_mm + offset
    print(f"  Z compensate: X={x_mm:.0f}mm Z_target={z_mm:.1f}mm offset={offset:.1f}mm → Z_ik={z_compensated:.1f}mm")
    return [target_xyz[0], target_xyz[1], z_compensated / 1000]

def _servo_dict_to_ik_seed(seed_servos):
    """Convert a {channel: servo_deg} dict into an IK initial_position vector.
    Returns None if no usable joints can be mapped."""
    if not seed_servos:
        return None
    seed = np.zeros(len(iris_chain.links))
    used = False
    for i, name in enumerate(JOINT_NAMES):
        cfg = SERVO_CONFIG[name]
        ch = cfg['ch']
        val = seed_servos.get(ch, seed_servos.get(str(ch)))
        if val is None:
            continue
        ik_deg = (float(val) - cfg['offset']) / cfg['dir']
        seed[i + 1] = np.radians(ik_deg)
        used = True
    return seed if used else None


def _servo_deg_to_ik_rad(joint_name, servo_deg):
    cfg = SERVO_CONFIG[joint_name]
    servo_deg = float(np.clip(float(servo_deg), cfg['min'], cfg['max']))
    ik_deg = (servo_deg - cfg['offset']) / cfg['dir']
    return np.radians(ik_deg)


def solve_ik(target_xyz, return_angles=False, seed_servos=None, wrist_pitch_deg=None):
    """Solve IK for target [x,y,z] in meters. Returns servo angles dict.
    Optional seed_servos={ch:deg} biases the solver toward a solution close to
    that current pose (used by smooth pattern mode in the visualizer).
    Optional wrist_pitch_deg biases IK toward a tool pitch orientation instead
    of overriding wrist pitch after the solve, which would move the tip."""
    # Adaptive seed based on target height
    target_z = target_xyz[2]
    target_x = target_xyz[0]

    # Calculate rough shoulder angle needed
    # Higher targets need more shoulder lift
    shoulder_hint = np.arctan2(target_z - 0.055, target_x) if target_x > 0 else np.radians(45)
    shoulder_hint = max(np.radians(10), shoulder_hint)

    # Try multiple seeds, pick the one with best score
    # Score = low error + shoulder is lifted (not down-then-fold)
    best_result = None
    best_ik_result = None
    best_score = float('inf')

    wrist_hint = None
    if wrist_pitch_deg is not None:
        wrist_hint = _servo_deg_to_ik_rad("wrist_pitch", wrist_pitch_deg)

    seeds = [
        # Seed 1: adaptive based on target
        {'shoulder': shoulder_hint, 'elbow': -shoulder_hint * 0.5, 'wrist': wrist_hint if wrist_hint is not None else 0},
        # Seed 2: shoulder very high
        {'shoulder': np.radians(80), 'elbow': np.radians(-60), 'wrist': wrist_hint if wrist_hint is not None else np.radians(-20)},
        # Seed 3: moderate
        {'shoulder': np.radians(45), 'elbow': np.radians(-30), 'wrist': wrist_hint if wrist_hint is not None else np.radians(10)},
        # Seed 4: straight up
        {'shoulder': np.radians(60), 'elbow': np.radians(-90), 'wrist': wrist_hint if wrist_hint is not None else np.radians(30)},
    ]

    warm_seed_vec = _servo_dict_to_ik_seed(seed_servos)

    # Build the list of initial_position vectors. When a warm seed is provided
    # we put it first so it has the best chance of being selected and we will
    # later add a closeness penalty for solutions that drift far from it.
    seed_vectors = []
    if warm_seed_vec is not None:
        seed_vectors.append(("warm", warm_seed_vec))
    for s in seeds:
        v = np.zeros(len(iris_chain.links))
        v[2] = s['shoulder']
        v[3] = s['elbow']
        v[4] = s.get('wrist', 0)
        seed_vectors.append(("preset", v))

    for tag, seed in seed_vectors:
        ik_result = iris_chain.inverse_kinematics(target_xyz, initial_position=seed)
        fk_result = iris_chain.forward_kinematics(ik_result)
        actual = fk_result[:3, 3]
        error = np.linalg.norm(np.array(target_xyz) - actual) * 1000

        # Check if servos are in valid range
        servo_angles = ik_to_servo(ik_result)
        all_valid = all(
            SERVO_CONFIG[name]['min'] <= servo_angles[SERVO_CONFIG[name]['ch']] <= SERVO_CONFIG[name]['max']
            for name in JOINT_NAMES
        )

        # Check elbow Z - penalize solutions where elbow goes below base
        frames = iris_chain.forward_kinematics(ik_result, full_kinematics=True)
        elbow_z = frames[3][2, 3]  # elbow link Z position

        # Score: lower is better
        # Penalize: high error, invalid servos, elbow below shoulder
        score = error
        if not all_valid:
            score += 1000
        if elbow_z < 0.05:  # elbow near or below ground
            score += 500
        if ik_result[2] < 0:  # shoulder angle negative (arm down in flipped URDF)
            score += 200

        # Smooth-pattern bias: when a warm seed is given, prefer solutions whose
        # servo angles are close to it. This kills branch-flipping in pattern mode
        # without affecting any caller that doesn't pass seed_servos.
        if warm_seed_vec is not None and seed_servos:
            joint_dist = 0.0
            for i, name in enumerate(JOINT_NAMES):
                cfg = SERVO_CONFIG[name]
                ch = cfg['ch']
                ref = seed_servos.get(ch, seed_servos.get(str(ch)))
                if ref is None:
                    continue
                joint_dist += abs(servo_angles[ch] - float(ref))
            # 1° of total servo travel costs 0.4mm of "virtual error". 30° of
            # branch flipping → 12mm penalty, which easily beats a 0mm IK error
            # tie between two valid branches.
            score += 0.4 * joint_dist

        if wrist_pitch_deg is not None:
            wp_ch = SERVO_CONFIG['wrist_pitch']['ch']
            wp_error = abs(float(servo_angles[wp_ch]) - float(wrist_pitch_deg))
            # Wrist pitch is a preference, not a reason to miss the target.
            # A high weight made far/high moves look unreachable even when the
            # arm could physically reach them with a slightly different pitch.
            score += 0.25 * wp_error

        if score < best_score:
            best_score = score
            best_result = (servo_angles, error, actual * 1000)
            best_ik_result = ik_result

    if return_angles:
        return best_result[0], best_result[1], best_result[2], best_ik_result
    return best_result

# ─── Camera streaming ───
camera_cap = None
camera_lock = threading.Lock()
camera_frame = None
camera_running = False

def start_camera():
    global camera_cap, camera_running, camera_frame
    if camera_running:
        return True
    try:
        camera_cap = cv2.VideoCapture(CAMERA_ID)
        if not camera_cap.isOpened():
            print(f"✗ Cannot open camera {CAMERA_ID}")
            return False
        camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        w = int(camera_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(camera_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera {CAMERA_ID} opened: {w}x{h}")
        if vision_uses_undistorted_pixels:
            if ensure_undistortion_maps(w, h):
                print("✓ Bridge camera stream uses undistorted pixels")
            else:
                print("⚠ Vision calibration expects undistorted pixels, but camera calibration is missing")
        camera_running = True

        def capture_loop():
            global camera_frame, camera_running
            while camera_running:
                ret, frame = camera_cap.read()
                if ret:
                    if vision_uses_undistorted_pixels and remap_map1 is not None and remap_map2 is not None:
                        frame = cv2.remap(frame, remap_map1, remap_map2, cv2.INTER_LINEAR)
                    with camera_lock:
                        camera_frame = frame
                else:
                    time.sleep(0.01)
            if camera_cap:
                camera_cap.release()

        t = threading.Thread(target=capture_loop, daemon=True)
        t.start()
        return True
    except Exception as e:
        print(f"✗ Camera error: {e}")
        return False

def stop_camera():
    global camera_running, camera_cap, camera_frame
    camera_running = False
    camera_frame = None
    print("✓ Camera stopped")

def get_camera_frame():
    with camera_lock:
        return camera_frame.copy() if camera_frame is not None else None

# ─── Serial ───
ser = None
current_angles = {}
target_angles = {}
move_lock = threading.Lock()
moving = False

def connect_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"✓ Serial connected: {SERIAL_PORT} @ {BAUD_RATE}")
        return True
    except Exception as e:
        print(f"✗ Serial error: {e}")
        return False

def send_raw(ch, angle):
    if ser and ser.is_open:
        cmd = f"ch {ch} {angle}\n"
        ser.write(cmd.encode())
        time.sleep(0.005)


def compensate_base_y_servo(angle):
    """Pull base/Y rotation 16% back toward the 75° center."""
    angle = float(angle)
    cfg = SERVO_CONFIG['base_rotation']
    compensated = BASE_Y_CENTER_DEG + (angle - BASE_Y_CENTER_DEG) * (1.0 - BASE_Y_INWARD_COMPENSATION)
    return float(np.clip(compensated, cfg['min'], cfg['max']))


def compensated_servo_targets(servo_angles):
    """Return the physical servo targets that set_target() will send."""
    out = dict(servo_angles)
    ch = SERVO_CONFIG['base_rotation']['ch']
    if ch in out:
        out[ch] = int(round(compensate_base_y_servo(out[ch])))
    return out


def smooth_move_worker():
    global moving, STEP_SIZE, STEP_DELAY
    while moving:
        any_moving = False
        with move_lock:
            for ch in list(target_angles.keys()):
                target = target_angles[ch]
                current = current_angles.get(ch, target)
                diff = abs(current - target)
                if diff < 0.5:
                    if current != target:
                        current_angles[ch] = target
                        send_raw(ch, int(target))
                    continue
                any_moving = True
                # Ease-in/ease-out: move faster in middle, slower near start/end
                # Use proportional speed based on remaining distance
                if diff < 5:
                    step = max(0.5, STEP_SIZE * 0.4)  # decelerate near target
                elif diff < 15:
                    step = STEP_SIZE * 0.7  # medium speed near target
                else:
                    step = STEP_SIZE  # full speed in middle
                if current < target:
                    current = min(current + step, target)
                else:
                    current = max(current - step, target)
                current_angles[ch] = current
                send_raw(ch, int(round(current)))
        time.sleep(STEP_DELAY if any_moving else 0.05)

def set_target(ch, angle):
    ch = int(ch)
    if ch == SERVO_CONFIG['base_rotation']['ch']:
        raw_angle = float(angle)
        angle = compensate_base_y_servo(raw_angle)
        if abs(angle - raw_angle) >= 0.05:
            print(f"  Base/Y inward comp: {raw_angle:.1f}° → {angle:.1f}°")
    with move_lock:
        if ch not in current_angles:
            current_angles[ch] = angle
        target_angles[ch] = angle

def servo_state_to_ik_angles(servo_state):
    """Convert the current servo state back into ikpy joint angles."""
    ik_angles = np.zeros(len(iris_chain.links))
    for i, name in enumerate(JOINT_NAMES):
        cfg = SERVO_CONFIG[name]
        ch = cfg['ch']
        servo_deg = float(servo_state.get(ch, cfg['offset']))
        ik_deg = (servo_deg - cfg['offset']) / cfg['dir']
        ik_angles[i + 1] = np.radians(ik_deg)
    return ik_angles

def current_fk_pose():
    """Return the robot pose implied by the currently tracked servo angles."""
    with move_lock:
        servo_state = {}
        for name in JOINT_NAMES:
            cfg = SERVO_CONFIG[name]
            ch = cfg['ch']
            servo_state[ch] = current_angles.get(ch, target_angles.get(ch, cfg['offset']))
        if 5 in current_angles or 5 in target_angles:
            servo_state[5] = current_angles.get(5, target_angles.get(5))
        if 0 in current_angles or 0 in target_angles:
            servo_state[0] = current_angles.get(0, target_angles.get(0))

    ik_angles = servo_state_to_ik_angles(servo_state)
    fk_result = iris_chain.forward_kinematics(ik_angles)
    actual_mm = fk_result[:3, 3] * 1000
    return actual_mm, servo_state, ik_angles

# ─── Robotics task state ───
robotics_cancel_flag = [False]  # mutable so threads can share
robotics_status = ["idle", ""]  # [status_text, last_action]
last_wrist_roll = [55]  # remember last wrist roll angle so it doesn't reset on lift

# ─── PATTERN RUNNER (smooth, constant-Z cartesian paths) ───
# The visualizer hands off the pattern descriptor and lets this thread tick at
# ~30ms with cartesian micro-steps. Each step is small enough that the
# smooth_move_worker teleports current→target (diff<0.5°), so the motion looks
# continuous instead of stuttering at HTTP latency.
pattern_state = {
    "running": False,
    "thread": None,
    "kind": "circle",
    "z_mm": 80.0,
    "cycle_ms": 8000,
    "wrist_roll_deg": None,
    "last_solution": None,  # {ch: deg} for warm-start
    "t": 0.0,
}
pattern_lock = threading.Lock()

def _pattern_point(kind, t, z_mm):
    cx, cy = 280.0, 0.0
    if kind == "line_x":
        tt = t * 2 if t < 0.5 else 2 - t * 2
        return (190 + tt * 180, 0.0, z_mm)
    if kind == "line_y":
        tt = t * 2 if t < 0.5 else 2 - t * 2
        return (280.0, -140 + tt * 280, z_mm)
    if kind == "circle":
        ang = t * 2 * np.pi
        return (cx + 90 * np.cos(ang), cy + 90 * np.sin(ang), z_mm)
    if kind == "square":
        r = 90.0
        corners = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        seg = int(t * 4) % 4
        local_t = (t * 4) % 1
        a = corners[seg]; b = corners[(seg + 1) % 4]
        return (a[0] + (b[0] - a[0]) * local_t, a[1] + (b[1] - a[1]) * local_t, z_mm)
    if kind == "zigzag":
        segs = 4
        xMin, xMax, yMin, yMax = 210.0, 350.0, -130.0, 130.0
        tt = t * 2 if t < 0.5 else 2 - t * 2
        seg = min(segs - 1, int(tt * segs))
        local_t = (tt * segs) - seg
        y = yMin + (yMax - yMin) * local_t if seg % 2 == 0 else yMax - (yMax - yMin) * local_t
        x = xMin + (xMax - xMin) * (seg / max(1, segs - 1))
        return (x, y, z_mm)
    return (cx, cy, z_mm)

def _pattern_loop():
    tick_s = 0.030  # 30ms — well above what serial can absorb, well under what feels jittery
    while True:
        with pattern_lock:
            if not pattern_state["running"]:
                return
            kind = pattern_state["kind"]
            z_mm = pattern_state["z_mm"]
            cycle_ms = max(2000, int(pattern_state["cycle_ms"]))
            t = pattern_state["t"]
            seed = pattern_state["last_solution"]
            wrist_roll_deg = pattern_state["wrist_roll_deg"]

        x_mm, y_mm, z_target = _pattern_point(kind, t, z_mm)
        target_m = [x_mm / 1000, y_mm / 1000, (z_target - PLATFORM_HEIGHT_MM) / 1000]
        try:
            servo_angles, error, _actual = solve_ik(target_m, seed_servos=seed)
        except Exception as e:
            print(f"  Pattern IK error: {e}")
            time.sleep(tick_s)
            continue

        # Honor a fixed wrist_roll override if requested; otherwise leave as solved.
        wr_cfg = SERVO_CONFIG['wrist_roll']
        if wrist_roll_deg is not None:
            servo_angles[wr_cfg['ch']] = int(round(np.clip(wrist_roll_deg, wr_cfg['min'], wr_cfg['max'])))
        servo_angles[5] = 180 - servo_angles[4]

        # Push targets — smooth_move_worker handles the actual serial output.
        for ch, angle in servo_angles.items():
            set_target(ch, angle)

        with pattern_lock:
            pattern_state["last_solution"] = dict(servo_angles)
            pattern_state["t"] = (t + tick_s * 1000.0 / cycle_ms) % 1.0

        time.sleep(tick_s)

def start_pattern(kind, z_mm, cycle_ms, wrist_roll_deg=None):
    with pattern_lock:
        if pattern_state["running"]:
            # Just update parameters live.
            pattern_state["kind"] = kind
            pattern_state["z_mm"] = float(z_mm)
            pattern_state["cycle_ms"] = int(cycle_ms)
            pattern_state["wrist_roll_deg"] = wrist_roll_deg
            return {"ok": True, "updated": True}
        pattern_state["kind"] = kind
        pattern_state["z_mm"] = float(z_mm)
        pattern_state["cycle_ms"] = int(cycle_ms)
        pattern_state["wrist_roll_deg"] = wrist_roll_deg
        pattern_state["t"] = 0.0
        # Warm the seed from current servo state so the first solve doesn't flip.
        pattern_state["last_solution"] = {ch: int(round(target_angles.get(ch, current_angles.get(ch, SERVO_CONFIG[name]['offset']))))
                                          for name in JOINT_NAMES for ch in [SERVO_CONFIG[name]['ch']]}
        pattern_state["running"] = True
        t = threading.Thread(target=_pattern_loop, daemon=True)
        pattern_state["thread"] = t
        t.start()
    print(f"✓ Pattern '{kind}' started @ Z={z_mm}mm cycle={cycle_ms}ms")
    return {"ok": True, "started": True}

def stop_pattern():
    with pattern_lock:
        was_running = pattern_state["running"]
        pattern_state["running"] = False
    if was_running:
        print("✓ Pattern stopped")
    return {"ok": True, "was_running": was_running}


def ps4_manual_status():
    """Return status for the optional PS4 manual controller subprocess."""
    global ps4_manual_process
    with ps4_manual_lock:
        proc = ps4_manual_process
        if proc is not None and proc.poll() is not None:
            ps4_manual_process = None
            proc = None
        return {
            "ok": True,
            "running": proc is not None,
            "pid": proc.pid if proc is not None else None,
            "script": PS4_MANUAL_SCRIPT_PATH,
            "python": PS4_MANUAL_PYTHON,
            "log": PS4_MANUAL_LOG_PATH,
        }


def start_ps4_manual_process():
    """Start the fresh PS4 controller script from the visualizer/bridge."""
    global ps4_manual_process
    with ps4_manual_lock:
        if ps4_manual_process is not None and ps4_manual_process.poll() is None:
            return ps4_manual_status()
        if not os.path.exists(PS4_MANUAL_SCRIPT_PATH):
            return {"ok": False, "running": False, "error": "iris_ps4_manual.py not found"}

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        log_file = open(PS4_MANUAL_LOG_PATH, "a", buffering=1)
        log_file.write("\n--- starting PS4 manual control ---\n")
        ps4_manual_process = subprocess.Popen(
            [PS4_MANUAL_PYTHON, PS4_MANUAL_SCRIPT_PATH, "--bridge", f"http://localhost:{HTTP_PORT}"],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return ps4_manual_status()


def stop_ps4_manual_process():
    """Stop the PS4 controller subprocess if it is running."""
    global ps4_manual_process
    with ps4_manual_lock:
        proc = ps4_manual_process
        if proc is None or proc.poll() is not None:
            ps4_manual_process = None
            return {"ok": True, "running": False}

        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
        ps4_manual_process = None
        return {"ok": True, "running": False}


def set_hello_pose(pose):
    for ch, angle in pose.items():
        set_target(ch, angle)


def run_hello_wave():
    global hello_running
    with hello_lock:
        if hello_running:
            return
        hello_running = True
    try:
        stop_pattern()
        print("✓ IRIS hello wave started")
        set_hello_pose(HELLO_POSE_A)
        time.sleep(1.4)

        low_wr, high_wr = HELLO_WRIST_ROLL_SWING
        for _ in range(HELLO_WAVE_REPETITIONS):
            steps = [
                (HELLO_POSE_A[6], low_wr, 0.28),
                (HELLO_POSE_A[6], high_wr, 0.28),
                (HELLO_POSE_B[6], low_wr, 0.65),
                (HELLO_POSE_B[6], high_wr, 0.28),
                (HELLO_POSE_B[6], low_wr, 0.28),
                (HELLO_POSE_A[6], high_wr, 0.65),
            ]
            for base_angle, wrist_roll, delay_s in steps:
                pose = dict(HELLO_POSE_A)
                pose[6] = base_angle
                pose[1] = wrist_roll
                set_hello_pose(pose)
                time.sleep(delay_s)

        set_hello_pose(HELLO_POSE_A)
        print("✓ IRIS hello wave done")
    finally:
        with hello_lock:
            hello_running = False


def start_hello_wave():
    with hello_lock:
        if hello_running:
            return {"ok": True, "running": True}
    t = threading.Thread(target=run_hello_wave, daemon=True)
    t.start()
    return {"ok": True, "running": True}


def hello_wave_status():
    with hello_lock:
        return {"ok": True, "running": hello_running}


def set_dance_pose(pose):
    for ch, angle in pose.items():
        set_target(ch, angle)


def sleep_dance(seconds):
    deadline = time.time() + max(0.0, seconds)
    while time.time() < deadline:
        if dance_stop_event.is_set():
            return False
        time.sleep(min(0.05, deadline - time.time()))
    return True


def apply_dance_speed(profile_name):
    global STEP_SIZE, STEP_DELAY
    step_size, step_delay = DANCE_SPEED_PROFILES.get(profile_name, DANCE_SPEED_PROFILES["medium"])
    STEP_SIZE = float(np.clip(step_size, DANCE_MIN_STEP_SIZE, DANCE_MAX_STEP_SIZE))
    STEP_DELAY = step_delay


def run_dance(seconds=DANCE_DEFAULT_DURATION_SECONDS, bpm=DANCE_DEFAULT_BPM):
    global dance_running, STEP_SIZE, STEP_DELAY
    with dance_lock:
        if dance_running:
            return
        dance_running = True
        dance_stop_event.clear()

    seconds = float(np.clip(seconds, 1.0, DANCE_MAX_DURATION_SECONDS))
    beat_s = 60.0 / max(40.0, min(220.0, float(bpm)))
    started = time.time()
    old_step_size = STEP_SIZE
    old_step_delay = STEP_DELAY
    try:
        stop_pattern()
        print(f"✓ IRIS dance started ({seconds:.1f}s @ {bpm:.1f} BPM)")
        apply_dance_speed("slow")
        set_dance_pose(DANCE_HOME_POSE)
        if not sleep_dance(beat_s * 2.0):
            return

        phase_index = -1
        step_index = 0
        while time.time() - started < seconds and not dance_stop_event.is_set():
            elapsed = time.time() - started
            phase = next((p for p in DANCE_PHASES if elapsed < p["until"]), DANCE_PHASES[-1])
            next_phase_index = DANCE_PHASES.index(phase)
            if next_phase_index != phase_index:
                phase_index = next_phase_index
                step_index = 0
                apply_dance_speed(phase.get("speed", "medium"))
                print(f"  Dance phase → {phase['name']}")

            pose_delta, beats = phase["steps"][step_index % len(phase["steps"])]
            pose = dict(DANCE_HOME_POSE)
            pose.update(pose_delta)
            set_dance_pose(pose)
            if not sleep_dance(beat_s * beats):
                break
            step_index += 1

        set_dance_pose(DANCE_HOME_POSE)
        print("✓ IRIS dance done")
    finally:
        STEP_SIZE = old_step_size
        STEP_DELAY = old_step_delay
        with dance_lock:
            dance_running = False
            dance_stop_event.clear()


def start_dance(seconds=DANCE_DEFAULT_DURATION_SECONDS, bpm=DANCE_DEFAULT_BPM):
    with dance_lock:
        if dance_running:
            return {"ok": True, "running": True}
    t = threading.Thread(target=run_dance, args=(seconds, bpm), daemon=True)
    t.start()
    return {"ok": True, "running": True, "duration_seconds": float(seconds), "bpm": float(bpm)}


def stop_dance():
    dance_stop_event.set()
    return {"ok": True, "running": False}


def dance_status():
    with dance_lock:
        return {"ok": True, "running": dance_running}

# ─── HTTP Server ───
class BridgeHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        global STEP_SIZE, STEP_DELAY
        parsed = urlparse(self.path)

        if parsed.path == "/ping":
            self.respond_json({
                "status": "ok",
                "serial": ser is not None and ser.is_open,
                "ikpy": True,
                "vision": vision_info.get("loaded", False)
            })
            return

        if parsed.path == "/vision/status":
            self.respond_json(vision_info)
            return

        if parsed.path == "/vision/reload":
            load_vision_calibration()
            load_vision_correction()
            self.respond_json(vision_info)
            return

        if parsed.path in ("/pose", "/fk"):
            try:
                actual_mm, servos, _ = current_fk_pose()
                self.respond_json({
                    "ok": True,
                    "actual": {
                        "x": round(float(actual_mm[0]), 1),
                        "y": round(float(actual_mm[1]), 1),
                        "z": round(float(actual_mm[2] + PLATFORM_HEIGHT_MM), 1),
                    },
                    "actual_base": {
                        "x": round(float(actual_mm[0]), 1),
                        "y": round(float(actual_mm[1]), 1),
                        "z": round(float(actual_mm[2]), 1),
                    },
                    "servos": {str(ch): round(float(angle), 1) for ch, angle in servos.items()},
                })
            except Exception as e:
                self.respond_json({"ok": False, "error": str(e)}, 500)
            return

        if parsed.path == "/servo":
            params = parse_qs(parsed.query)
            ch = int(params.get("ch", [0])[0])
            angle = int(params.get("angle", [90])[0])
            set_target(ch, angle)
            sent_angle = compensate_base_y_servo(angle) if ch == SERVO_CONFIG['base_rotation']['ch'] else angle
            self.respond_json({"ok": True, "ch": ch, "angle": angle, "sent_angle": round(float(sent_angle), 1)})
            return

        if parsed.path == "/speed":
            params = parse_qs(parsed.query)
            if "step" in params:
                STEP_SIZE = max(0.5, min(10, float(params["step"][0])))
            if "delay" in params:
                STEP_DELAY = max(0.005, min(0.1, float(params["delay"][0]) / 1000))
            self.respond_json({"ok": True, "step": STEP_SIZE, "delay_ms": STEP_DELAY * 1000})
            return

        # ─── Gripper INSTANT (bypass smooth mover) ───
        if parsed.path == "/gripper":
            params = parse_qs(parsed.query)
            angle = int(params.get("angle", [0])[0])
            angle = max(0, min(70, angle))  # clamp 0-70
            send_raw(0, angle)
            with move_lock:
                current_angles[0] = angle
                target_angles[0] = angle
            print(f"  Gripper INSTANT → {angle}°")
            self.respond_json({"ok": True, "ch": 0, "angle": angle, "instant": True})
            return

        # ─── Camera control ───
        if parsed.path == "/camera/start":
            ok = start_camera()
            self.respond_json({"ok": ok})
            return

        if parsed.path == "/camera/stop":
            stop_camera()
            self.respond_json({"ok": True})
            return

        if parsed.path == "/camera/status":
            self.respond_json({"running": camera_running})
            return

        if parsed.path == "/gamepad/status":
            self.respond_json(ps4_manual_status())
            return

        if parsed.path == "/hello/status":
            self.respond_json(hello_wave_status())
            return

        if parsed.path == "/dance/status":
            self.respond_json(dance_status())
            return

        # ─── MJPEG video feed ───
        if parsed.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while camera_running:
                    frame = get_camera_frame()
                    if frame is not None:
                        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b"\r\n")
                    time.sleep(0.033)  # ~30fps
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        # ─── Single frame snapshot ───
        if parsed.path == "/camera/snapshot":
            frame = get_camera_frame()
            if frame is not None:
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(jpeg.tobytes())
            else:
                self.respond_json({"error": "No camera frame"}, 503)
            return

        # ─── Single frame as base64 for Gemini Live ───
        if parsed.path == "/camera/frame_b64":
            frame = get_camera_frame()
            if frame is not None:
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"data": b64, "mimeType": "image/jpeg"}).encode())
            else:
                self.respond_json({"error": "No camera frame"}, 503)
            return

        if parsed.path == "/media/iris_dance.mp3":
            if os.path.exists(DANCE_AUDIO_PATH):
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(os.path.getsize(DANCE_AUDIO_PATH)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(DANCE_AUDIO_PATH, "rb") as f:
                    self.wfile.write(f.read())
                return
            self.respond_json({"error": "iris_dance.mp3 not found"}, 404)
            return

        # ─── Serve HTML visualizer ───
        if parsed.path == "/" or parsed.path == "/index.html":
            html_path = os.path.join(SCRIPT_DIR, "iris_visualizer.html")
            if os.path.exists(html_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            self.respond_json({"error": "iris_visualizer.html not found"}, 404)
            return

        # ─── Serve IRIS Live interface ───
        if parsed.path == "/live":
            html_path = os.path.join(SCRIPT_DIR, "iris_live.html")
            if os.path.exists(html_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            self.respond_json({"error": "iris_live.html not found"}, 404)
            return

        self.respond_json({"error": "unknown"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()

        if parsed.path == "/gamepad/start":
            self.respond_json(start_ps4_manual_process())
            return

        if parsed.path == "/gamepad/stop":
            self.respond_json(stop_ps4_manual_process())
            return

        if parsed.path == "/hello":
            self.respond_json(start_hello_wave())
            return

        if parsed.path == "/dance/start":
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {}
            seconds = float(data.get("duration_seconds", 30.0))
            bpm = float(data.get("bpm", DANCE_DEFAULT_BPM))
            self.respond_json(start_dance(seconds, bpm))
            return

        if parsed.path == "/dance/stop":
            self.respond_json(stop_dance())
            return

        if parsed.path == "/servos":
            try:
                data = json.loads(body)
                sent = {}
                for ch_str, angle in data.items():
                    ch = int(ch_str)
                    set_target(ch, int(angle))
                    sent[ch] = round(float(compensate_base_y_servo(angle) if ch == SERVO_CONFIG['base_rotation']['ch'] else angle), 1)
                self.respond_json({"ok": True, "sent": sent})
            except Exception as e:
                self.respond_json({"error": str(e)}, 400)
            return

        # ─── IK ENDPOINT ───
        if parsed.path == "/vision/pixel_to_robot":
            try:
                data = json.loads(body)
                px, py = data["px"], data["py"]
                z = float(data.get("z", 0.0))
                apply_corr = bool(data.get("apply_correction", True))
                result = pixel_to_robot(px, py, z, apply_corr)
                if result is None:
                    self.respond_json({"error": "No vision calibration loaded"}, 400)
                    return
                rx, ry = result
                self.respond_json({
                    "ok": True,
                    "pixel": {"x": px, "y": py},
                    "robot": {"x": round(rx, 2), "y": round(ry, 2), "z": z},
                    "correction_applied": apply_corr and vision_correction is not None,
                })
            except Exception as e:
                self.respond_json({"error": str(e)}, 400)
            return

        # ─── IK ENDPOINT ───
        if parsed.path == "/ik":
            try:
                data = json.loads(body)
                # Input in mm, convert to meters for ikpy
                # Subtract platform height: robot base is elevated above table
                z_adjusted = data["z"] - PLATFORM_HEIGHT_MM
                target_m = [data["x"] / 1000, data["y"] / 1000, z_adjusted / 1000]
                seed_servos = data.get("seed_servos")  # optional {ch: deg} — smooth/warm-start
                wrist_pitch_deg = data.get("wrist_pitch_deg", None)
                servo_angles, error, actual_mm, ik_result = solve_ik(
                    target_m,
                    return_angles=True,
                    seed_servos=seed_servos,
                    wrist_pitch_deg=wrist_pitch_deg,
                )

                wp_cfg = SERVO_CONFIG['wrist_pitch']
                wp_angle = servo_angles[wp_cfg['ch']]

                wrist_roll_deg = data.get("wrist_roll_deg", None)
                wr_cfg = SERVO_CONFIG['wrist_roll']
                if wrist_roll_deg is not None:
                    wr_angle = int(round(np.clip(wrist_roll_deg, wr_cfg['min'], wr_cfg['max'])))
                else:
                    wr_angle = wr_cfg['offset']
                servo_angles[wr_cfg['ch']] = wr_angle
                servo_angles[5] = 180 - servo_angles[4]
                sent_servo_angles = compensated_servo_targets(servo_angles)

                print(f"  IK: target=({data['x']}, {data['y']}, {data['z']})mm "
                      f"→ actual=({actual_mm[0]:.1f}, {actual_mm[1]:.1f}, {actual_mm[2]:.1f})mm "
                      f"err={error:.1f}mm wp={wp_angle}° wr={wr_angle}°")
                print(f"  Servos raw: {servo_angles}")
                print(f"  Servos sent: {sent_servo_angles}")

                # Send to robot
                if data.get("send", False):
                    for ch, angle in servo_angles.items():
                        set_target(ch, angle)

                self.respond_json({
                    "ok": True,
                    "servos": sent_servo_angles,
                    "ik_servos_raw": servo_angles,
                    "error_mm": round(error, 2),
                    "actual": {
                        "x": round(actual_mm[0], 1),
                        "y": round(actual_mm[1], 1),
                        "z": round(actual_mm[2], 1)
                    }
                })
            except Exception as e:
                print(f"  IK error: {e}")
                self.respond_json({"error": str(e)}, 400)
            return

        # ─── SMOOTH PATTERN STREAMING ───
        if parsed.path == "/pattern/start":
            try:
                data = json.loads(body)
                kind = data.get("kind", "circle")
                z_mm = float(data.get("z_mm", 80))
                cycle_ms = int(data.get("cycle_ms", 8000))
                wrist_roll_deg = data.get("wrist_roll_deg", None)
                result = start_pattern(kind, z_mm, cycle_ms, wrist_roll_deg)
                self.respond_json(result)
            except Exception as e:
                self.respond_json({"ok": False, "error": str(e)}, 400)
            return

        if parsed.path == "/pattern/stop":
            self.respond_json(stop_pattern())
            return

        # ─── VISION CLICK → IK ENDPOINT ───
        if parsed.path == "/vision/click":
            try:
                data = json.loads(body)
                px, py = data["px"], data["py"]
                pick_z = data.get("z", vision_info.get("pick_z", 100))
                send = data.get("send", False)

                use_z_plane = bool(data.get("use_z_plane", False))
                xy_z = pick_z if use_z_plane else 0.0

                apply_corr = bool(data.get("apply_correction", True))

                result = pixel_to_robot(px, py, xy_z, apply_corr)
                if result is None:
                    self.respond_json({"error": "No vision calibration loaded"}, 400)
                    return

                rx, ry = result
                plane_label = f"Z={xy_z:.0f}" if use_z_plane else "table"
                print(f"  Vision: px({px},{py}) [{plane_label}] → robot({rx:.1f},{ry:.1f},{pick_z})mm")

                # Solve IK
                target_m = [rx / 1000, ry / 1000, pick_z / 1000]
                servo_angles, error, actual_mm = solve_ik(target_m)

                # ── Post-IK wrist roll override ──
                wrist_roll_deg = data.get("wrist_roll_deg", None)
                wr_cfg = SERVO_CONFIG['wrist_roll']
                if wrist_roll_deg is not None:
                    wr_angle = int(round(np.clip(wrist_roll_deg, wr_cfg['min'], wr_cfg['max'])))
                else:
                    wr_angle = wr_cfg['offset']
                servo_angles[wr_cfg['ch']] = wr_angle
                servo_angles[5] = 180 - servo_angles[4]
                sent_servo_angles = compensated_servo_targets(servo_angles)

                print(f"    → IK err={error:.1f}mm wr={wr_angle}° servos={sent_servo_angles} raw={servo_angles}")

                if send:
                    for ch, angle in servo_angles.items():
                        set_target(ch, angle)

                self.respond_json({
                    "ok": True,
                    "pixel": {"x": px, "y": py},
                    "robot": {"x": round(rx, 1), "y": round(ry, 1), "z": pick_z},
                    "xy_plane_z": round(xy_z, 1),
                    "correction_applied": apply_corr and vision_correction is not None,
                    "servos": sent_servo_angles,
                    "ik_servos_raw": servo_angles,
                    "error_mm": round(error, 2),
                    "actual": {
                        "x": round(actual_mm[0], 1),
                        "y": round(actual_mm[1], 1),
                        "z": round(actual_mm[2], 1)
                    }
                })
            except Exception as e:
                print(f"  Vision click error: {e}")
                self.respond_json({"error": str(e)}, 400)
            return

        # ─── GEMINI ROBOTICS ER ENDPOINT ───
        if parsed.path == "/gemini":
            try:
                data = json.loads(body)
                prompt = data.get("prompt", "")
                pick_z = data.get("z", vision_info.get("pick_z", 100))
                send = data.get("send", False)
                auto_pick = data.get("auto_pick", False)

                # Get camera snapshot
                frame = get_camera_frame()
                if frame is None:
                    self.respond_json({"error": "Camera not running. Start camera first."}, 400)
                    return

                img_h, img_w = frame.shape[:2]

                # Build the full prompt with coordinate format instructions
                full_prompt = f"""{prompt}
The answer should follow the json format: [{{"point": [y, x], "label": <label>}}, ...].
The points are in [y, x] format normalized to 0-1000."""

                # Encode frame as JPEG
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                image_bytes = jpeg.tobytes()

                print(f"  Gemini: prompt='{prompt}' img={img_w}x{img_h}")

                # Query Gemini
                gemini_result = query_gemini(full_prompt, image_bytes)

                if "error" in gemini_result and "ok" not in gemini_result:
                    print(f"    Gemini error: {gemini_result['error']}")
                    self.respond_json(gemini_result, 400)
                    return

                # Convert normalized points to pixel coords
                raw_points = gemini_result.get("points", [])
                if isinstance(raw_points, list):
                    pixel_points = gemini_points_to_pixels(raw_points, img_w, img_h)
                else:
                    pixel_points = []

                print(f"    Gemini found {len(pixel_points)} objects")

                # Convert to robot coords via homography
                robot_points = []
                for pp in pixel_points:
                    robot_xy = pixel_to_robot(pp["px"], pp["py"])
                    if robot_xy:
                        rx, ry = robot_xy
                        robot_points.append({
                            "label": pp["label"],
                            "pixel": {"x": pp["px"], "y": pp["py"]},
                            "robot": {"x": round(rx, 1), "y": round(ry, 1), "z": pick_z},
                            "norm": pp["norm"]
                        })
                        print(f"      {pp['label']}: px({pp['px']},{pp['py']}) → robot({rx:.1f},{ry:.1f})")

                # Auto-pick: move to first detected object
                ik_result = None
                if auto_pick and len(robot_points) > 0 and send:
                    target = robot_points[0]
                    target_m = [target["robot"]["x"] / 1000, target["robot"]["y"] / 1000, pick_z / 1000]
                    servo_angles, error, actual_mm = solve_ik(target_m)
                    sent_servo_angles = compensated_servo_targets(servo_angles)
                    for ch, angle in servo_angles.items():
                        set_target(ch, angle)
                    ik_result = {
                        "target": target["label"],
                        "servos": sent_servo_angles,
                        "ik_servos_raw": servo_angles,
                        "error_mm": round(error, 2)
                    }
                    print(f"    → Moving to '{target['label']}' err={error:.1f}mm")

                self.respond_json({
                    "ok": True,
                    "objects": robot_points,
                    "raw": gemini_result.get("raw", ""),
                    "ik": ik_result
                })

            except Exception as e:
                print(f"  Gemini error: {e}")
                import traceback
                traceback.print_exc()
                self.respond_json({"error": str(e)}, 400)
            return

        # ─── ROBOTICS DESCRIBE SCENE (for Live API to ask what's on table) ───
        if parsed.path == "/robotics/describe":
            try:
                data = json.loads(body) if body else {}
                question = data.get("question", "Descrie tot ce vezi pe masa de lucru. Fii concis.")

                frame = get_camera_frame()
                if frame is None:
                    self.respond_json({"error": "Camera not running"}, 400)
                    return

                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

                request_body = {
                    "contents": [{"parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                        {"text": question}
                    ]}],
                    "generationConfig": {"temperature": 0.5, "thinkingConfig": {"thinkingBudget": 512}}
                }

                import urllib.request as urlreq
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
                req = urlreq.Request(url, data=json.dumps(request_body).encode('utf-8'),
                                     headers={"Content-Type": "application/json"}, method="POST")
                with urlreq.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read().decode('utf-8'))

                text = ""
                for candidate in result.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "text" in part:
                            text += part["text"]

                self.respond_json({"ok": True, "description": text})

            except Exception as e:
                self.respond_json({"error": str(e)}, 400)
            return

        # ─── ROBOTICS CANCEL ───
        if parsed.path == "/robotics/cancel":
            robotics_cancel_flag[0] = True
            print("  🛑 Robotics cancel requested!")
            # Send arm home immediately
            send_raw(0, 0)  # open gripper
            with move_lock:
                current_angles[0] = 0
                target_angles[0] = 0
            time.sleep(0.5)
            target_m = [0.2, 0.0, 0.25]
            servo_angles, error, actual_mm = solve_ik(target_m)
            for ch, angle in servo_angles.items():
                set_target(ch, angle)
            last_wrist_roll[0] = 55  # reset wrist roll
            # Wait a bit so the running task thread sees the cancel flag
            time.sleep(1.0)
            self.respond_json({"ok": True, "cancelled": True})
            return

        # ─── ROBOTICS STATUS ───
        if parsed.path == "/robotics/status":
            self.respond_json({
                "busy": robotics_status[0] != "idle",
                "status": robotics_status[0],
                "last_action": robotics_status[1],
            })
            return

        # ─── ROBOTICS EXECUTE ENDPOINT (called by Live API) ───
        # Live API sends a task description, Robotics ER does the multi-turn execution
        if parsed.path == "/robotics/execute":
            try:
                data = json.loads(body)
                task = data.get("task", "")

                if not GEMINI_API_KEY:
                    self.respond_json({"error": "GEMINI_API_KEY not set"}, 400)
                    return

                # If a task is already running, cancel it first
                if robotics_status[0] != "idle":
                    print(f"  ⚠ Previous task still running, cancelling...")
                    robotics_cancel_flag[0] = True
                    # Send arm home
                    send_raw(0, 0)
                    with move_lock:
                        current_angles[0] = 0
                        target_angles[0] = 0
                    # Wait for old task to see cancel flag and stop
                    for _ in range(20):  # max 2 seconds
                        time.sleep(0.1)
                        if robotics_status[0] == "idle":
                            break

                # Reset for new task
                robotics_cancel_flag[0] = False
                robotics_status[0] = f"Executing: {task[:60]}"
                robotics_status[1] = "starting"
                last_wrist_roll[0] = 55  # fresh wrist roll for new task

                # Setup SSE streaming
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                def send_sse(event, payload):
                    line = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    self.wfile.write(line.encode('utf-8'))
                    self.wfile.flush()

                # Build initial message with a fresh camera frame for context
                initial_parts = [{"text": task}]
                frame = get_camera_frame()
                if frame is not None:
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    img_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                    initial_parts.insert(0, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}})

                history = [{"role": "user", "parts": initial_parts}]

                url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
                agent_tools = AGENT_TOOL_DECLARATIONS
                max_turns = 30
                actions_log = []
                object_missing_count = 0
                failed_motion_count = 0

                print(f"  🤖 Robotics execute: '{task}'")

                for turn in range(max_turns):
                    # Check cancel flag
                    if robotics_cancel_flag[0]:
                        send_sse("done", {"summary": "Task anulat de utilizator.", "actions": actions_log, "cancelled": True})
                        robotics_status[0] = "idle"
                        robotics_status[1] = "cancelled"
                        print(f"  🛑 Robotics cancelled at turn {turn}")
                        return

                    request_body = {
                        "systemInstruction": {"parts": [{"text": AGENT_SYSTEM_PROMPT}]},
                        "contents": history,
                        "tools": [{"functionDeclarations": agent_tools}],
                        "generationConfig": {"temperature": 0.1}
                    }

                    import urllib.request as urlreq
                    req = urlreq.Request(
                        url,
                        data=json.dumps(request_body).encode('utf-8'),
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )

                    try:
                        with urlreq.urlopen(req, timeout=30) as resp:
                            result = json.loads(resp.read().decode('utf-8'))
                    except Exception as e:
                        send_sse("error", {"text": f"Robotics API error: {e}"})
                        send_sse("done", {"summary": f"Error: {e}", "actions": actions_log})
                        return

                    candidates = result.get("candidates", [])
                    if not candidates:
                        send_sse("error", {"text": "No response from Robotics"})
                        send_sse("done", {"summary": "No response", "actions": actions_log})
                        return

                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])

                    function_calls = [p for p in parts if "functionCall" in p]
                    text_parts = [p.get("text", "") for p in parts if "text" in p]
                    intermediate_text = " ".join(text_parts).strip()

                    if intermediate_text and function_calls:
                        send_sse("thinking", {"text": intermediate_text})

                    history.append(content)

                    if not function_calls:
                        # Final response from Robotics
                        robotics_status[0] = "idle"
                        robotics_status[1] = "done"
                        send_sse("done", {"summary": intermediate_text, "actions": actions_log})
                        print(f"  🤖 Robotics done: {intermediate_text[:100]}")
                        return

                    # Execute function calls
                    function_responses = []
                    for fc_part in function_calls:
                        # Check cancel between actions
                        if robotics_cancel_flag[0]:
                            robotics_status[0] = "idle"
                            send_sse("done", {"summary": "Task anulat.", "actions": actions_log, "cancelled": True})
                            print(f"  🛑 Cancelled during function execution")
                            return

                        fc = fc_part["functionCall"]
                        fn_name = fc["name"]
                        fn_args = fc.get("args", {})

                        robotics_status[1] = f"{fn_name}({json.dumps(fn_args, ensure_ascii=False)[:50]})"
                        send_sse("action", {"name": fn_name, "args": fn_args})
                        print(f"    ⚡ {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:80]})")

                        fn_result = agent_execute_function(fn_name, fn_args)

                        actions_log.append({"name": fn_name, "args": fn_args, "result": fn_result})
                        send_sse("action_result", {"name": fn_name, "result": fn_result})

                        result_error = str(fn_result.get("error", "")).lower() if isinstance(fn_result, dict) else ""
                        object_missing = (
                            fn_name in ("detect_objects", "side_grasp_object")
                            and (
                                fn_result.get("success") is False
                                or fn_result.get("count") == 0
                                or "not detected" in result_error
                                or "object not detected" in result_error
                            )
                        )
                        if object_missing:
                            object_missing_count += 1
                        elif fn_name in ("detect_objects", "side_grasp_object") and fn_result.get("success") is not False:
                            object_missing_count = 0

                        motion_failed = (
                            fn_name in ("move_arm", "move_to_safe_height", "side_grasp_object")
                            and fn_result.get("success") is False
                            and ("ik error" in result_error or "unreachable" in result_error)
                        )
                        if motion_failed:
                            failed_motion_count += 1
                        elif fn_result.get("success") is not False:
                            failed_motion_count = 0

                        if object_missing_count >= 2:
                            summary = "Mă opresc: nu mai detectez obiectul cerut, deci probabil a fost luat/mutat sau nu mai este în cadru."
                            robotics_status[0] = "idle"
                            robotics_status[1] = "object missing"
                            send_sse("done", {"summary": summary, "actions": actions_log, "stopped": True, "reason": "object_missing"})
                            print(f"  🛑 Robotics stopped: object missing after {object_missing_count} attempts")
                            return

                        if failed_motion_count >= 3:
                            summary = "Mă opresc: mișcările cerute nu sunt rezolvate stabil de IK în poziția asta."
                            robotics_status[0] = "idle"
                            robotics_status[1] = "motion failed"
                            send_sse("done", {"summary": summary, "actions": actions_log, "stopped": True, "reason": "motion_failed"})
                            print(f"  🛑 Robotics stopped: repeated motion failure")
                            return

                        function_responses.append({
                            "functionResponse": {
                                "name": fn_name,
                                "response": fn_result
                            }
                        })

                    history.append({"role": "user", "parts": function_responses})

                robotics_status[0] = "idle"
                robotics_status[1] = "max turns"
                send_sse("done", {"summary": "Mă opresc: am atins limita de pași pentru task-ul curent.", "actions": actions_log, "stopped": True, "reason": "max_turns"})

            except Exception as e:
                print(f"  Robotics execute error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    self.respond_json({"error": str(e)}, 400)
                except:
                    pass
            return

        # ─── AGENT CHAT ENDPOINT (SSE streaming) ───
        if parsed.path == "/agent/chat":
            try:
                data = json.loads(body)
                user_message = data.get("message", "")
                history = data.get("history", [])

                if not GEMINI_API_KEY:
                    self.respond_json({"error": "GEMINI_API_KEY not set"}, 400)
                    return

                # Setup SSE
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                def send_sse(event, payload):
                    line = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    self.wfile.write(line.encode('utf-8'))
                    self.wfile.flush()

                # Add user message to history
                history.append({"role": "user", "parts": [{"text": user_message}]})

                url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

                agent_tools = AGENT_TOOL_DECLARATIONS
                max_turns = 15

                for turn in range(max_turns):
                    request_body = {
                        "systemInstruction": {"parts": [{"text": AGENT_SYSTEM_PROMPT}]},
                        "contents": history,
                        "tools": [{"functionDeclarations": agent_tools}],
                        "generationConfig": {"temperature": 0.7}
                    }

                    import urllib.request as urlreq
                    req = urlreq.Request(
                        url,
                        data=json.dumps(request_body).encode('utf-8'),
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )

                    try:
                        with urlreq.urlopen(req, timeout=30) as resp:
                            result = json.loads(resp.read().decode('utf-8'))
                    except Exception as e:
                        send_sse("error", {"text": f"Gemini API error: {e}"})
                        send_sse("done", {})
                        return

                    candidates = result.get("candidates", [])
                    if not candidates:
                        send_sse("error", {"text": "No response from Gemini"})
                        send_sse("done", {})
                        return

                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])

                    function_calls = [p for p in parts if "functionCall" in p]
                    text_parts = [p.get("text", "") for p in parts if "text" in p]

                    # Stream any intermediate text
                    intermediate_text = " ".join(text_parts).strip()
                    if intermediate_text and function_calls:
                        send_sse("thinking", {"text": intermediate_text})

                    history.append(content)

                    if not function_calls:
                        # Final response
                        send_sse("response", {"text": intermediate_text, "history": history})
                        send_sse("done", {})
                        return

                    # Execute function calls
                    function_responses = []
                    for fc_part in function_calls:
                        fc = fc_part["functionCall"]
                        fn_name = fc["name"]
                        fn_args = fc.get("args", {})

                        send_sse("action", {"name": fn_name, "args": fn_args})
                        print(f"  Agent ⚡ {fn_name}({json.dumps(fn_args, ensure_ascii=False)})")

                        fn_result = agent_execute_function(fn_name, fn_args)

                        send_sse("action_result", {"name": fn_name, "result": fn_result})

                        function_responses.append({
                            "functionResponse": {
                                "name": fn_name,
                                "response": fn_result
                            }
                        })

                    history.append({"role": "user", "parts": function_responses})

                send_sse("response", {"text": "Max function call rounds reached", "history": history})
                send_sse("done", {})

            except Exception as e:
                print(f"  Agent chat error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    self.respond_json({"error": str(e)}, 400)
                except:
                    pass
            return

        self.respond_json({"error": "unknown"}, 404)

    def respond_json(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  IRIS Bridge v6.0 — Agent + Voice + Vision        ║")
    print("║  Browser → Agent/Gemini → IK → Serial → ESP32     ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    if not connect_serial():
        print("⚠ Running without serial (test mode)\n")

    moving = True
    t = threading.Thread(target=smooth_move_worker, daemon=True)
    t.start()
    print("✓ Smooth mover started")
    print(f"● http://localhost:{HTTP_PORT}")
    print(f"  GET  /              — visualizer UI")
    print(f"  GET  /live          — IRIS Live voice interface")
    print(f"  GET  /video_feed    — MJPEG camera stream")
    print(f"  POST /ik            — solve IK")
    print(f"  POST /gemini        — Gemini ER vision query")
    print(f"  POST /robotics/execute — Robotics task execution (SSE)")
    print(f"  POST /robotics/describe — Robotics scene description")
    print(f"  POST /agent/chat    — Agent chat (SSE streaming)")
    print(f"  POST /vision/click  — pixel → robot → IK")
    print(f"  POST /hello         — audience greeting wave")
    print(f"  POST /dance/start   — 30s music dance routine")
    print(f"  POST /dance/stop    — stop music dance routine")
    print(f"  POST /gamepad/start — start PS4 manual controller")
    print(f"  POST /gamepad/stop  — stop PS4 manual controller")
    print(f"  GET  /gripper       — instant gripper (0=open, 70=closed)")
    print("Press Ctrl+C to stop\n")

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("0.0.0.0", HTTP_PORT), BridgeHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n● Shutting down...")
        moving = False
        stop_ps4_manual_process()
        stop_camera()
        if ser and ser.is_open:
            ser.close()
        server.server_close()
        print("✓ Done")
