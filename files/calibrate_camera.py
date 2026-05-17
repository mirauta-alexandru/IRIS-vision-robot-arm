#!/usr/bin/env python3
"""
IRIS Camera Calibration with ChArUco Board (v2 — improved edge accuracy)

Changes from v1:
  - Coverage heatmap: guides you to capture ALL zones (center + edges)
  - Per-zone error reporting (center vs edges)
  - Higher min captures (15) for better lens model
  - Optimal calibration flags for edge distortion
  - Visual feedback on which zones still need captures

1. Print charuco_board.pdf at 100% scale on A4 Landscape
2. Stick to rigid cardboard
3. Mount camera fixed above robot workspace
4. Run this script
5. Show ChArUco board in 15+ positions — COVER ALL ZONES (watch heatmap!)
6. SPACE = capture, Q = calibrate, ESC = quit
"""

import cv2
import numpy as np
import json
import os
import sys

# ─── CONFIG ───
CAMERA_ID = 0
BOARD_SQUARES = (7, 5)
SQUARE_SIZE = 35.0       # mm
MARKER_SIZE = 25.0       # mm
MIN_CAPTURES = 15        # ↑ from 12 — more captures = better lens model
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Coverage grid: divide image into 3x3 zones, track captures per zone
COVERAGE_GRID = (3, 3)

# ─── Setup ChArUco ───
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard(BOARD_SQUARES, SQUARE_SIZE / 1000, MARKER_SIZE / 1000, dictionary)
detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
charuco_detector = cv2.aruco.CharucoDetector(board)

all_charuco_corners = []
all_charuco_ids = []
all_obj_points = []
captures = 0
img_size = None

# Coverage tracking
coverage_counts = np.zeros(COVERAGE_GRID, dtype=int)

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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
img_size = (w, h)


def get_coverage_zone(corners, w, h):
    """Return set of (row, col) zones that these corners cover."""
    zones = set()
    cell_w = w / COVERAGE_GRID[1]
    cell_h = h / COVERAGE_GRID[0]
    for pt in corners.reshape(-1, 2):
        col = min(int(pt[0] / cell_w), COVERAGE_GRID[1] - 1)
        row = min(int(pt[1] / cell_h), COVERAGE_GRID[0] - 1)
        zones.add((row, col))
    return zones


def draw_coverage_overlay(display, w, h):
    """Draw coverage heatmap overlay — red=uncovered, green=well-covered."""
    cell_w = w // COVERAGE_GRID[1]
    cell_h = h // COVERAGE_GRID[0]
    overlay = display.copy()

    for r in range(COVERAGE_GRID[0]):
        for c in range(COVERAGE_GRID[1]):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            count = coverage_counts[r, c]

            if count == 0:
                color = (0, 0, 180)   # Red — needs captures
                alpha = 0.25
            elif count < 2:
                color = (0, 140, 255)  # Orange — needs more
                alpha = 0.15
            elif count < 4:
                color = (0, 200, 200)  # Yellow — OK
                alpha = 0.08
            else:
                color = (0, 180, 0)    # Green — good
                alpha = 0.05

            sub = overlay[y1:y2, x1:x2]
            rect = np.full_like(sub, color, dtype=np.uint8)
            cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)

            # Zone label
            label = str(count)
            cv2.putText(overlay, label,
                        (x1 + cell_w // 2 - 8, y1 + cell_h // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Grid lines
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 80, 80), 1)

    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)


def get_zone_label(r, c):
    """Human-readable zone label."""
    row_names = ["Top", "Mid", "Bot"]
    col_names = ["Left", "Center", "Right"]
    return f"{row_names[r]}-{col_names[c]}"


def suggest_next_zone():
    """Suggest which zone needs more captures."""
    min_count = coverage_counts.min()
    for r in range(COVERAGE_GRID[0]):
        for c in range(COVERAGE_GRID[1]):
            if coverage_counts[r, c] == min_count:
                return get_zone_label(r, c), min_count
    return "All", min_count


print(f"Camera {CAMERA_ID}: {w}x{h}")
print()
print("═" * 60)
print("  IRIS ChArUco Camera Calibration v2")
print("═" * 60)
print(f"  Board: {BOARD_SQUARES[0]}x{BOARD_SQUARES[1]} squares")
print(f"  Square: {SQUARE_SIZE}mm | Marker: {MARKER_SIZE}mm")
print(f"  Min captures: {MIN_CAPTURES}")
print()
print("  ★ IMPORTANT: Cover ALL zones! Watch the heatmap overlay.")
print("    Move the board to edges & corners of the frame too!")
print()
print("  SPACE = capture | Q = calibrate | ESC = quit")
print("═" * 60)

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
        zone_name, zone_count = suggest_next_zone()
        status_text = f"DETECTED {num_corners} corners — Press SPACE | Need more: {zone_name}"
    else:
        status_color = (0, 0, 255)
        status_text = "Show ChArUco board to camera"

    # Coverage heatmap overlay
    draw_coverage_overlay(display, w, h)

    # UI header
    cv2.rectangle(display, (0, 0), (w, 45), (20, 20, 20), -1)
    min_zone = coverage_counts.min()
    coverage_status = "GOOD" if min_zone >= 2 else f"Need edges! (min zone={min_zone})"
    cv2.putText(display, f"Captures: {captures}/{MIN_CAPTURES} | Coverage: {coverage_status}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    # Footer
    cv2.rectangle(display, (0, h - 40), (w, h), (20, 20, 20), -1)
    cv2.putText(display, status_text,
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

    # Progress bar
    progress = min(1.0, captures / MIN_CAPTURES)
    cv2.rectangle(display, (10, 38), (int(10 + (w - 20) * progress), 43), (0, 200, 255), -1)

    cv2.imshow("IRIS ChArUco Calibration v2", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and num_corners > 3:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

        # Track coverage
        zones = get_coverage_zone(charuco_corners, w, h)
        for (r, c) in zones:
            coverage_counts[r, c] += 1

        captures += 1
        zone_name, _ = suggest_next_zone()
        print(f"  ✓ Capture {captures} ({num_corners} corners) — Next: move board to {zone_name}")

        # Flash
        flash = display.copy()
        cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), 8)
        cv2.imshow("IRIS ChArUco Calibration v2", flash)
        cv2.waitKey(150)

    elif key == ord('q'):
        if captures >= MIN_CAPTURES:
            # Check coverage
            if coverage_counts.min() < 1:
                uncovered = []
                for r in range(COVERAGE_GRID[0]):
                    for c in range(COVERAGE_GRID[1]):
                        if coverage_counts[r, c] < 1:
                            uncovered.append(get_zone_label(r, c))
                print(f"  ⚠ Uncovered zones: {', '.join(uncovered)}")
                print(f"  Press Q again to calibrate anyway, or capture more.")
                # Allow pressing Q again to force
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord('q'):
                    break
            else:
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

# Build object/image point pairs
obj_points = []
img_points = []
for corners, ids in zip(all_charuco_corners, all_charuco_ids):
    if corners is not None and len(corners) >= 4:
        obj_pts = board.getChessboardCorners()[ids.flatten()]
        obj_points.append(obj_pts.astype(np.float32))
        img_points.append(corners.astype(np.float32))

# Calibrate with extra distortion coefficients for better edge modeling
# CALIB_RATIONAL_MODEL adds k4,k5,k6 for heavy lens distortion
# Try standard first, then rational if edge error is high
print("  Pass 1: Standard model (k1-k3 + p1-p2)...")
ret_std, cam_mtx_std, dist_std, rvecs_std, tvecs_std = cv2.calibrateCamera(
    obj_points, img_points, img_size, None, None
)
print(f"    Reprojection error: {ret_std:.4f} px")

print("  Pass 2: Rational model (k1-k6 + p1-p2)...")
ret_rat, cam_mtx_rat, dist_rat, rvecs_rat, tvecs_rat = cv2.calibrateCamera(
    obj_points, img_points, img_size, None, None,
    flags=cv2.CALIB_RATIONAL_MODEL
)
print(f"    Reprojection error: {ret_rat:.4f} px")

# Pick the better model
if ret_rat < ret_std * 0.9:  # Rational must be meaningfully better
    print(f"  → Using RATIONAL model (k1-k6) — {((ret_std - ret_rat) / ret_std * 100):.1f}% better")
    ret = ret_rat
    camera_matrix = cam_mtx_rat
    dist_coeffs = dist_rat
    model_type = "rational"
else:
    print(f"  → Using STANDARD model (k1-k3) — rational not significantly better")
    ret = ret_std
    camera_matrix = cam_mtx_std
    dist_coeffs = dist_std
    model_type = "standard"

print()
print(f"  Final reprojection error: {ret:.4f} pixels")
print(f"  (< 0.3 = excellent, < 0.5 = very good, < 1.0 = good)")

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

print()
print(f"  Camera Matrix:")
print(f"    fx = {fx:.2f}")
print(f"    fy = {fy:.2f}")
print(f"    cx = {cx:.2f}")
print(f"    cy = {cy:.2f}")
print()
print(f"  Distortion ({model_type}): [{', '.join(f'{d:.6f}' for d in dist_coeffs.ravel())}]")

# ─── Per-zone error analysis ───
print()
print("  ── Per-zone reprojection error ──")
zone_errors = [[[] for _ in range(COVERAGE_GRID[1])] for _ in range(COVERAGE_GRID[0])]
cell_w = w / COVERAGE_GRID[1]
cell_h = h / COVERAGE_GRID[0]

for i, (obj_pts, img_pts, rvec, tvec) in enumerate(
    zip(obj_points, img_points,
        rvecs_rat if model_type == "rational" else rvecs_std,
        tvecs_rat if model_type == "rational" else tvecs_std)):
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    actual = img_pts.reshape(-1, 2)

    for j in range(len(actual)):
        err = np.linalg.norm(projected[j] - actual[j])
        col = min(int(actual[j][0] / cell_w), COVERAGE_GRID[1] - 1)
        row = min(int(actual[j][1] / cell_h), COVERAGE_GRID[0] - 1)
        zone_errors[row][col].append(err)

for r in range(COVERAGE_GRID[0]):
    row_str = "  "
    for c in range(COVERAGE_GRID[1]):
        errs = zone_errors[r][c]
        if errs:
            mean_e = np.mean(errs)
            row_str += f"  {get_zone_label(r, c)}: {mean_e:.3f}px ({len(errs)}pts)"
        else:
            row_str += f"  {get_zone_label(r, c)}: NO DATA"
    print(row_str)

# ─── Compute and save undistortion maps ───
print()
print("  Computing undistortion maps...")
new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, img_size, alpha=0, newImgSize=img_size
)
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, new_cam_mtx, img_size, cv2.CV_32FC1
)
print(f"  ✓ Undistortion maps ready ({img_size[0]}x{img_size[1]})")

# ─── Save ───
calib_data = {
    "camera_id": CAMERA_ID,
    "resolution": list(img_size),
    "camera_matrix": camera_matrix.tolist(),
    "new_camera_matrix": new_cam_mtx.tolist(),
    "dist_coeffs": dist_coeffs.tolist(),
    "reprojection_error": ret,
    "model_type": model_type,
    "num_captures": captures,
    "square_size_mm": SQUARE_SIZE,
    "marker_size_mm": MARKER_SIZE,
    "board_size": list(BOARD_SQUARES),
    "dictionary": "DICT_4X4_50",
    "coverage_counts": coverage_counts.tolist(),
    "roi": list(roi),
}

calib_path = os.path.join(SAVE_DIR, "camera_calibration.json")
with open(calib_path, "w") as f:
    json.dump(calib_data, f, indent=2)

np_path = os.path.join(SAVE_DIR, "camera_calibration.npz")
np.savez(np_path,
    camera_matrix=camera_matrix,
    new_camera_matrix=new_cam_mtx,
    dist_coeffs=dist_coeffs,
    map1=map1,
    map2=map2,
)

print()
print(f"  ✓ Saved: {calib_path}")
print(f"  ✓ Saved: {np_path} (includes undistortion maps)")
print()
print("  Coverage summary:")
for r in range(COVERAGE_GRID[0]):
    for c in range(COVERAGE_GRID[1]):
        status = "✓" if coverage_counts[r, c] >= 2 else "⚠" if coverage_counts[r, c] >= 1 else "✗"
        print(f"    {status} {get_zone_label(r, c)}: {coverage_counts[r, c]} captures")
print()
print("Next: run iris_vision_v3.py — it will auto-undistort using these maps.")
