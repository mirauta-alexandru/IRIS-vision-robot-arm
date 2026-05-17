#!/usr/bin/env python3
"""
IRIS Vision Correction Calibrator

Uses an ArUco marker mounted on the gripper to learn a residual correction map
on top of the normal vision calibration.

Assumption for the current IRIS gripper marker:
  - Marker ID 21, DICT_4X4_50
  - Marker center is 120 mm behind the gripper tip, toward the robot base
  - Marker and gripper tip are on the same Z plane during calibration

Workflow:
  1. Start iris_bridge.py
  2. Make sure iris_vision_v3.py calibration is saved
  3. Run this script
  4. It moves the robot, detects marker ID21, saves vision_correction.json
"""

import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import requests


BRIDGE_URL = "http://localhost:8765"
SAVE_DIR = Path(__file__).resolve().parent
CORRECTION_PATH = SAVE_DIR / "vision_correction.json"
DEBUG_IMAGE_PATH = SAVE_DIR / "vision_correction_debug.png"

MARKER_ID = 21
MARKER_DICTIONARY = cv2.aruco.DICT_4X4_50

# Marker center -> gripper tip distance, in the gripper's forward/radial direction.
MARKER_BEHIND_TIP_MM = 120.0

# Keep this close to the actual object pick plane, while still safely above the table.
CALIB_Z_MM = 45.0
WRIST_ROLL_DEG = 55
MOVE_SETTLE_SECONDS = 2.6
DETECTIONS_PER_POINT = 5

# Tip targets. The script synthesizes the tip position from the marker position,
# so these are the coordinates we want the gripper tip to be corrected at.
TIP_TARGETS = [
    (150, -70), (150, 0), (150, 70),
    (210, -90), (210, 0), (210, 90),
    (270, -90), (270, 0), (270, 90),
    (325, -65), (325, 0), (325, 65),
]


def post(path, payload, timeout=8):
    r = requests.post(f"{BRIDGE_URL}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get(path, timeout=8):
    r = requests.get(f"{BRIDGE_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r


def radial_unit(x_mm, y_mm):
    norm = math.hypot(x_mm, y_mm)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    return np.array([x_mm / norm, y_mm / norm], dtype=np.float64)


def expected_tip_from_marker(marker_xy, true_tip_xy):
    """Marker is behind the tip toward base, so tip = marker + radial * distance."""
    u = radial_unit(true_tip_xy[0], true_tip_xy[1])
    return np.array(marker_xy, dtype=np.float64) + u * MARKER_BEHIND_TIP_MM


def detect_marker_center():
    dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICTIONARY)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    centers = []
    last_frame = None
    last_corners = None

    for _ in range(40):
        resp = get("/camera/snapshot", timeout=5)
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            time.sleep(0.1)
            continue

        corners, ids, _ = detector.detectMarkers(frame)
        last_frame = frame
        last_corners = corners

        if ids is not None:
            for idx, marker_id in enumerate(ids.flatten()):
                if int(marker_id) == MARKER_ID:
                    pts = corners[idx].reshape(4, 2)
                    center = pts.mean(axis=0)
                    centers.append(center)
                    if len(centers) >= DETECTIONS_PER_POINT:
                        avg = np.mean(np.array(centers), axis=0)
                        return float(avg[0]), float(avg[1]), frame, corners, ids

        time.sleep(0.12)

    return None, None, last_frame, last_corners, None


def raw_pixel_to_robot(px, py, z_mm):
    data = post("/vision/pixel_to_robot", {
        "px": float(px),
        "py": float(py),
        "z": float(z_mm),
        "apply_correction": False,
    })
    robot = data["robot"]
    return np.array([robot["x"], robot["y"]], dtype=np.float64)


def move_tip(x_mm, y_mm):
    data = post("/ik", {
        "x": float(x_mm),
        "y": float(y_mm),
        "z": float(CALIB_Z_MM),
        "wrist_roll_deg": WRIST_ROLL_DEG,
        "send": True,
    }, timeout=10)
    if not data.get("ok"):
        raise RuntimeError(data)
    actual = data.get("actual", {})
    time.sleep(MOVE_SETTLE_SECONDS)
    return np.array([actual.get("x", x_mm), actual.get("y", y_mm)], dtype=np.float64)


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


def main():
    print()
    print("═" * 62)
    print("  IRIS Vision Correction Calibrator")
    print("═" * 62)
    print(f"  Marker: ArUco 4x4_50 ID {MARKER_ID}")
    print(f"  Offset: marker is {MARKER_BEHIND_TIP_MM:.0f}mm behind gripper tip")
    print(f"  Z plane: {CALIB_Z_MM:.0f}mm")
    print()

    status = get("/vision/status").json()
    if not status.get("loaded"):
        raise SystemExit("✗ Bridge has no vision calibration loaded. Run iris_vision_v3.py and /vision/reload first.")

    print(f"✓ Vision loaded: {status.get('method', 'unknown')}")
    get("/camera/start")
    time.sleep(1.0)

    samples = []
    debug_frame = None

    for idx, (tx, ty) in enumerate(TIP_TARGETS, 1):
        print()
        print(f"[{idx:02d}/{len(TIP_TARGETS)}] Move tip to X={tx:.0f} Y={ty:.0f} Z={CALIB_Z_MM:.0f}")
        true_tip = move_tip(tx, ty)
        print(f"    actual tip: ({true_tip[0]:.1f}, {true_tip[1]:.1f})")

        px, py, frame, corners, ids = detect_marker_center()
        if px is None:
            print("    ✗ marker not detected, skipping point")
            continue

        debug_frame = frame
        observed_marker = raw_pixel_to_robot(px, py, CALIB_Z_MM)
        observed_tip = expected_tip_from_marker(observed_marker, true_tip)
        residual = true_tip - observed_tip
        err = float(np.linalg.norm(residual))

        print(f"    marker px: ({px:.1f}, {py:.1f})")
        print(f"    observed tip: ({observed_tip[0]:.1f}, {observed_tip[1]:.1f})")
        print(f"    residual: dx={residual[0]:+.1f} dy={residual[1]:+.1f} err={err:.1f}mm")

        samples.append({
            "target_tip_xy": [float(tx), float(ty)],
            "true_tip_xy": [float(true_tip[0]), float(true_tip[1])],
            "marker_pixel": [float(px), float(py)],
            "observed_marker_xy": [float(observed_marker[0]), float(observed_marker[1])],
            "observed_tip_xy": [float(observed_tip[0]), float(observed_tip[1])],
            "residual_xy": [float(residual[0]), float(residual[1])],
            "error_mm": err,
        })

    if len(samples) < 4:
        raise SystemExit(f"✗ Only {len(samples)} samples captured. Need at least 4.")

    before_errors = [s["error_mm"] for s in samples]
    after_errors = []
    for i, sample in enumerate(samples):
        corrected = idw_correct(sample["observed_tip_xy"], samples, exclude_index=i)
        true_tip = np.array(sample["true_tip_xy"], dtype=np.float64)
        after_errors.append(float(np.linalg.norm(corrected - true_tip)))

    save_data = {
        "method": "idw_residual_field",
        "marker_dictionary": "DICT_4X4_50",
        "marker_id": MARKER_ID,
        "marker_behind_tip_mm": MARKER_BEHIND_TIP_MM,
        "calibration_z_mm": CALIB_Z_MM,
        "assumption": "marker_center_is_behind_gripper_tip_toward_robot_base",
        "power": 2.0,
        "max_radius_mm": 220.0,
        "avg_error_before": float(np.mean(before_errors)),
        "max_error_before": float(np.max(before_errors)),
        "avg_error_after": float(np.mean(after_errors)),
        "max_error_after": float(np.max(after_errors)),
        "points": samples,
    }

    with open(CORRECTION_PATH, "w") as f:
        json.dump(save_data, f, indent=2)

    print()
    print("─" * 62)
    print(f"✓ Saved correction: {CORRECTION_PATH}")
    print(f"  Before: avg={save_data['avg_error_before']:.1f}mm max={save_data['max_error_before']:.1f}mm")
    print(f"  LOO estimate after: avg={save_data['avg_error_after']:.1f}mm max={save_data['max_error_after']:.1f}mm")

    try:
        get("/vision/reload")
        print("✓ Bridge reloaded vision calibration + correction")
    except Exception as e:
        print(f"⚠ Could not reload bridge automatically: {e}")

    if debug_frame is not None:
        cv2.imwrite(str(DEBUG_IMAGE_PATH), debug_frame)
        print(f"✓ Last debug frame: {DEBUG_IMAGE_PATH}")

    print("═" * 62)


if __name__ == "__main__":
    main()
