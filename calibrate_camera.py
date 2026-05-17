#!/usr/bin/env python3
"""
IRIS Camera Calibration with ChArUco Board
1. Print charuco_board.pdf at 100% scale on A4 Landscape
2. Stick to rigid cardboard
3. Mount camera fixed above robot workspace  
4. Run this script
5. Show ChArUco board in 15-20 different positions/angles
6. SPACE = capture, Q = calibrate, ESC = quit
"""

import cv2
import numpy as np
import json
import os
import sys

# ─── CONFIG ───
CAMERA_ID = 1           # 0=MacBook, 1=iPhone
BOARD_SQUARES = (7, 5)  # Columns x Rows of squares
SQUARE_SIZE = 35.0      # mm
MARKER_SIZE = 25.0      # mm
MIN_CAPTURES = 12
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Setup ChArUco ───
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard(BOARD_SQUARES, SQUARE_SIZE / 1000, MARKER_SIZE / 1000, dictionary)
detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
charuco_detector = cv2.aruco.CharucoDetector(board)

all_charuco_corners = []
all_charuco_ids = []
captures = 0
img_size = None

# ─── Open camera ───
print(f"Opening camera {CAMERA_ID}...")
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"Cannot open camera {CAMERA_ID}. Trying camera 0...")
    CAMERA_ID = 0
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("No camera available!")
        sys.exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
img_size = (w, h)
print(f"Camera {CAMERA_ID}: {w}x{h}")
print()
print("═" * 55)
print("  IRIS ChArUco Camera Calibration")
print("═" * 55)
print(f"  Board: {BOARD_SQUARES[0]}x{BOARD_SQUARES[1]} squares")
print(f"  Square: {SQUARE_SIZE}mm | Marker: {MARKER_SIZE}mm")
print(f"  Min captures: {MIN_CAPTURES}")
print()
print("  SPACE = capture | Q = calibrate | ESC = quit")
print("═" * 55)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display = frame.copy()

    # Detect ChArUco
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
    
    num_corners = 0
    if charuco_corners is not None and len(charuco_corners) > 3:
        num_corners = len(charuco_corners)
        cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids, (0, 255, 0))
        
        if marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
        
        status_color = (0, 255, 0)
        status_text = f"DETECTED {num_corners} corners - Press SPACE"
    else:
        status_color = (0, 0, 255)
        status_text = "Show ChArUco board to camera"

    # Draw UI
    cv2.rectangle(display, (0, 0), (w, 45), (20, 20, 20), -1)
    cv2.putText(display, f"IRIS ChArUco Calibration | Captures: {captures}/{MIN_CAPTURES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.rectangle(display, (0, h - 40), (w, h), (20, 20, 20), -1)
    cv2.putText(display, status_text,
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # Progress bar
    progress = min(1.0, captures / MIN_CAPTURES)
    cv2.rectangle(display, (10, 38), (int(10 + (w - 20) * progress), 43), (0, 200, 255), -1)

    cv2.imshow("IRIS ChArUco Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and num_corners > 3:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        captures += 1
        print(f"  ✓ Capture {captures} ({num_corners} corners)")

        # Flash
        flash = display.copy()
        cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), 8)
        cv2.imshow("IRIS ChArUco Calibration", flash)
        cv2.waitKey(150)

    elif key == ord('q'):
        if captures >= MIN_CAPTURES:
            break
        else:
            print(f"  Need at least {MIN_CAPTURES} captures! (have {captures})")

    elif key == 27:
        print("Cancelled.")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

cap.release()
cv2.destroyAllWindows()

# ─── Calibrate ───
print()
print("Calibrating with ChArUco...")

# Convert CharUco corners to calibrateCamera format
obj_points = []
img_points = []

for corners, ids in zip(all_charuco_corners, all_charuco_ids):
    if corners is not None and len(corners) >= 4:
        obj_pts = board.getChessboardCorners()[ids.flatten()]
        obj_points.append(obj_pts.astype(np.float32))
        img_points.append(corners.astype(np.float32))

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, img_size, None, None
)

print(f"  Reprojection error: {ret:.4f} pixels")
print(f"  (< 0.5 = excellent, < 1.0 = good)")
print()

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

print(f"  Camera Matrix:")
print(f"    fx = {fx:.2f}")
print(f"    fy = {fy:.2f}")
print(f"    cx = {cx:.2f}")
print(f"    cy = {cy:.2f}")
print()
print(f"  Distortion: [{', '.join(f'{d:.6f}' for d in dist_coeffs.ravel())}]")

# ─── Save ───
calib_data = {
    "camera_id": CAMERA_ID,
    "resolution": list(img_size),
    "camera_matrix": camera_matrix.tolist(),
    "dist_coeffs": dist_coeffs.tolist(),
    "reprojection_error": ret,
    "num_captures": captures,
    "square_size_mm": SQUARE_SIZE,
    "marker_size_mm": MARKER_SIZE,
    "board_size": list(BOARD_SQUARES),
    "dictionary": "DICT_4X4_50",
}

calib_path = os.path.join(SAVE_DIR, "camera_calibration.json")
with open(calib_path, "w") as f:
    json.dump(calib_data, f, indent=2)

np_path = os.path.join(SAVE_DIR, "camera_calibration.npz")
np.savez(np_path,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
)

print()
print(f"  ✓ Saved: {calib_path}")
print(f"  ✓ Saved: {np_path}")
print()
print("Next step: place ChArUco board at known position relative to robot base")
print("to calibrate camera-to-robot transformation.")
