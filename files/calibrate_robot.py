#!/usr/bin/env python3
"""
IRIS Camera-to-Robot Calibration
Board is placed at known position relative to robot base.
This script detects the board, computes the camera-to-robot transform,
and saves it for use in the vision pipeline.

Prerequisites:
- camera_calibration.json (from calibrate_camera.py)
- ChArUco board at known position relative to robot base
"""

import cv2
import numpy as np
import json
import os
import sys

# ─── CONFIG ───
CAMERA_ID = 0
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Board config (must match calibrate_camera.py)
BOARD_SQUARES = (7, 5)
SQUARE_SIZE = 35.0   # mm
MARKER_SIZE = 25.0   # mm

# Board position relative to robot base (mm)
# The origin of the board (first corner) is at this position
# relative to the robot base center
print()
print("═" * 55)
print("  Board offset from robot base")
print("  Measure from robot base center to board first corner")
print("═" * 55)
try:
    BOARD_OFFSET_X = float(input("  X offset (mm, forward from base) [85]: ").strip() or "85")
    BOARD_OFFSET_Y = float(input("  Y offset (mm, lateral, 0=centered) [0]: ").strip() or "0")
    BOARD_OFFSET_Z = float(input("  Z offset (mm, vertical, 0=same height) [0]: ").strip() or "0")
except ValueError:
    print("Invalid input, using defaults (85, 0, 0)")
    BOARD_OFFSET_X = 85.0
    BOARD_OFFSET_Y = 0.0
    BOARD_OFFSET_Z = 0.0

# ─── Load camera calibration ───
calib_path = os.path.join(SAVE_DIR, "camera_calibration.json")
if not os.path.exists(calib_path):
    print("ERROR: camera_calibration.json not found!")
    print("Run calibrate_camera.py first.")
    sys.exit(1)

with open(calib_path) as f:
    calib = json.load(f)

camera_matrix = np.array(calib["camera_matrix"])
dist_coeffs = np.array(calib["dist_coeffs"])
print(f"✓ Loaded calibration (error: {calib['reprojection_error']:.4f}px)")

# ─── Setup ChArUco ───
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard(BOARD_SQUARES, SQUARE_SIZE / 1000, MARKER_SIZE / 1000, dictionary)
charuco_detector = cv2.aruco.CharucoDetector(board)

# ─── Open camera ───
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    CAMERA_ID = 0
    cap = cv2.VideoCapture(CAMERA_ID)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✓ Camera {CAMERA_ID}: {w}x{h}")
print()
print()
print("═" * 55)
print("  IRIS Camera-to-Robot Calibration")
print("═" * 55)
print(f"  Board offset: X={BOARD_OFFSET_X}mm  Y={BOARD_OFFSET_Y}mm  Z={BOARD_OFFSET_Z}mm")
print()
print("  SPACE = capture & compute transform")
print("  Q/ESC = quit")
print("═" * 55)

best_transform = None
captures = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display = frame.copy()

    # Detect ChArUco
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    num_corners = 0
    if charuco_corners is not None and len(charuco_corners) >= 4:
        num_corners = len(charuco_corners)
        cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids, (0, 255, 0))
        if marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

        # Estimate board pose
        obj_points = board.getChessboardCorners()[charuco_ids.flatten()]
        success, rvec, tvec = cv2.solvePnP(
            obj_points, charuco_corners, camera_matrix, dist_coeffs
        )

        if success:
            # Draw axes on board
            cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

            # Board position in camera frame (meters)
            board_in_cam = tvec.flatten() * 1000  # to mm

            # Rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            status_text = f"DETECTED | Board at ({board_in_cam[0]:.0f}, {board_in_cam[1]:.0f}, {board_in_cam[2]:.0f})mm from camera"
            status_color = (0, 255, 0)
        else:
            status_text = "Board detected but pose failed"
            status_color = (0, 165, 255)
    else:
        status_text = "Show ChArUco board - keep it still"
        status_color = (0, 0, 255)
        success = False

    # UI
    cv2.rectangle(display, (0, 0), (w, 45), (20, 20, 20), -1)
    cv2.putText(display, f"Camera-Robot Calibration | Captures: {captures}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.rectangle(display, (0, h - 40), (w, h), (20, 20, 20), -1)
    cv2.putText(display, status_text,
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    cv2.imshow("Camera-Robot Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and success:
        captures += 1

        # ─── Compute camera-to-robot transform ───
        # We know:
        # - Board origin in camera frame: R, tvec
        # - Board origin in robot frame: (BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_OFFSET_Z)

        # Transform: point_robot = R_cam2robot @ point_camera + t_cam2robot
        # 
        # Board origin in camera frame: t_board_cam = tvec
        # Board origin in robot frame: t_board_robot = [X, Y, Z] offset
        #
        # For a point on the board at local position p_local:
        #   p_camera = R @ p_local + tvec
        #   p_robot = p_local * 1000 + [BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_OFFSET_Z]
        #
        # So: p_robot = (R^-1 @ (p_camera - tvec)) * 1000 + board_offset
        # Or equivalently: p_robot = R_cam2robot @ p_camera + t_cam2robot

        R_board2cam = R  # rotation from board frame to camera frame
        t_board2cam = tvec.flatten()  # translation from board frame to camera frame (meters)

        # Camera to board: inverse
        R_cam2board = R_board2cam.T
        t_cam2board = -R_cam2board @ t_board2cam

        # Board to robot: just a translation (board is aligned with robot axes)
        # board_origin_in_robot = [BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_OFFSET_Z] in mm
        # board coordinates are in meters, robot in mm
        board_offset_m = np.array([BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_OFFSET_Z]) / 1000

        # Full transform: camera -> robot
        # p_robot_m = R_cam2board @ p_cam + t_cam2board + board_offset_m
        # But we need to be careful about axis conventions
        # In robot frame: X=forward, Y=left, Z=up
        # In camera frame: X=right, Y=down, Z=forward (OpenCV convention)

        # Combined transform
        R_cam2robot = R_cam2board.copy()
        t_cam2robot = (t_cam2board + board_offset_m) * 1000  # back to mm

        best_transform = {
            "R_cam2robot": R_cam2robot.tolist(),
            "t_cam2robot": t_cam2robot.tolist(),
            "R_board2cam": R_board2cam.tolist(),
            "t_board2cam": (t_board2cam * 1000).tolist(),  # mm
            "rvec": rvec.flatten().tolist(),
            "tvec": (tvec.flatten() * 1000).tolist(),  # mm
            "board_offset_mm": [BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_OFFSET_Z],
        }

        print(f"  ✓ Capture {captures}")
        print(f"    Board in camera: ({board_in_cam[0]:.1f}, {board_in_cam[1]:.1f}, {board_in_cam[2]:.1f})mm")
        print(f"    Transform computed")

        # Flash
        flash = display.copy()
        cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), 8)
        cv2.imshow("Camera-Robot Calibration", flash)
        cv2.waitKey(150)

    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

if best_transform is None:
    print("No transform captured!")
    sys.exit(1)

# ─── Save transform ───
transform_data = {
    "camera_id": CAMERA_ID,
    "camera_matrix": camera_matrix.tolist(),
    "dist_coeffs": dist_coeffs.tolist(),
    **best_transform,
}

transform_path = os.path.join(SAVE_DIR, "camera_robot_transform.json")
with open(transform_path, "w") as f:
    json.dump(transform_data, f, indent=2)

print()
print(f"✓ Saved: {transform_path}")
print()
print("Calibration complete!")
print("You can now use this transform to convert camera pixels to robot coordinates.")
