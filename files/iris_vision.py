#!/usr/bin/env python3
"""
IRIS Vision v2 — ChArUco Auto-Calibration + Pick (with undistortion)

Changes from v1:
  - Applies lens undistortion on every frame (fixes edge accuracy!)
  - Multi-capture: accumulate board positions across workspace → better homography
  - Per-zone error display (center vs edges)
  - LMEDS instead of RANSAC for homography
  - Undistortion maps precomputed for speed

Keys:
  SPACE = capture board position (accumulates for multi-point calibration)
  C = compute homography from all captures
  F = flip axis mapping (rotate 90°)
  T = test mode (show coords without moving)
  G = toggle grid overlay
  E = show per-zone error map
  S = save calibration
  +/- = adjust pick Z
  R = recalibrate (clear all captures)
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

# ─── CONFIG ───
CAMERA_ID = 0
BRIDGE_URL = "http://localhost:8765"
PICK_Z = 100
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
calib_path = os.path.join(SAVE_DIR, "camera_calibration.json")
npz_path = os.path.join(SAVE_DIR, "camera_calibration.npz")
transform_path = os.path.join(SAVE_DIR, "charuco_calibration.json")

# Board config
BOARD_SQUARES = (7, 5)
SQUARE_SIZE = 35.0        # mm
MARKER_SIZE = 25.0        # mm

# Workspace bounds (mm)
WORKSPACE_X_MIN = 80
WORKSPACE_X_MAX = 350
WORKSPACE_Y_MIN = -120
WORKSPACE_Y_MAX = 120

# ─── State ───
homography = None
mode = "detect"  # detect → pick
last_click = None
show_grid = True
show_error_map = False
test_mode = False
axis_mapping = 0
board_offset_x = 85.0
board_offset_y = 0.0
avg_error = None
num_points_used = 0
input_q = queue.Queue()

# Multi-capture accumulation
all_px_points = []      # list of (N,2) arrays — pixel coords (undistorted)
all_robot_points = []   # list of (N,2) arrays — robot coords
capture_count = 0

# Undistortion
use_undistort = False
remap_map1 = None
remap_map2 = None
undist_cam_matrix = None  # new camera matrix after undistortion

# Error zones
ZONE_GRID = (3, 3)
zone_errors_mm = None  # filled after calibration


# ─── Axis mapping configs ───
AXIS_CONFIGS = [
    {"label": "Board cols → Robot +X, rows → Robot +Y",  "x_sign": 1, "y_sign": 1,  "swap": False},
    {"label": "Board cols → Robot +Y, rows → Robot -X",  "x_sign": -1, "y_sign": 1, "swap": True},
    {"label": "Board cols → Robot -X, rows → Robot -Y",  "x_sign": -1, "y_sign": -1, "swap": False},
    {"label": "Board cols → Robot -Y, rows → Robot +X",  "x_sign": 1, "y_sign": -1,  "swap": True},
]


def get_board_corners_robot(charuco_ids, axis_cfg):
    """Convert ChArUco corner IDs to robot-frame XY coordinates (mm)."""
    num_cols = BOARD_SQUARES[0] - 1

    robot_points = []
    for cid in charuco_ids.flatten():
        col = cid % num_cols
        row = cid // num_cols

        local_x = col * SQUARE_SIZE
        local_y = row * SQUARE_SIZE

        cfg = axis_cfg
        if cfg["swap"]:
            rx = cfg["x_sign"] * local_y + board_offset_x
            ry = cfg["y_sign"] * local_x + board_offset_y
        else:
            rx = cfg["x_sign"] * local_x + board_offset_x
            ry = cfg["y_sign"] * local_y + board_offset_y

        robot_points.append([rx, ry])

    return np.array(robot_points, dtype=np.float64)


def add_capture(charuco_corners, charuco_ids):
    """Add a board detection to the multi-capture buffer."""
    global capture_count

    cfg = AXIS_CONFIGS[axis_mapping]
    px_pts = charuco_corners.reshape(-1, 2).astype(np.float64)
    robot_pts = get_board_corners_robot(charuco_ids, cfg)

    all_px_points.append(px_pts)
    all_robot_points.append(robot_pts)
    capture_count += 1

    print(f"  ✓ Capture {capture_count}: {len(px_pts)} points")
    print(f"    Robot range: X=[{robot_pts[:, 0].min():.0f}..{robot_pts[:, 0].max():.0f}] "
          f"Y=[{robot_pts[:, 1].min():.0f}..{robot_pts[:, 1].max():.0f}]")

    return True


def compute_calibration_multi():
    """Compute homography from ALL accumulated captures."""
    global homography, avg_error, num_points_used, zone_errors_mm

    if capture_count == 0:
        print("  ✗ No captures! Press SPACE first.")
        return False

    cfg = AXIS_CONFIGS[axis_mapping]

    # Merge all captures
    all_px = np.vstack(all_px_points)
    all_rb = np.vstack(all_robot_points)
    num_points_used = len(all_px)

    print(f"\n  ── Multi-Capture Calibration ──")
    print(f"  Captures: {capture_count}")
    print(f"  Total points: {num_points_used}")
    print(f"  Axis: {cfg['label']}")
    print(f"  Offset: X={board_offset_x}mm Y={board_offset_y}mm")

    # Use LMEDS — no threshold needed, robust to outliers
    homography, mask = cv2.findHomography(all_px, all_rb, cv2.LMEDS)

    if homography is None:
        print("  ✗ Homography failed!")
        return False

    # Compute per-point errors
    errors = []
    for i in range(len(all_px)):
        h = homography @ [all_px[i][0], all_px[i][1], 1.0]
        predicted = h[:2] / h[2]
        actual = all_rb[i]
        err = np.sqrt((predicted[0] - actual[0]) ** 2 + (predicted[1] - actual[1]) ** 2)
        errors.append(err)

    errors = np.array(errors)
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    median_error = np.median(errors)

    if mask is not None:
        inliers = mask.ravel().sum()
        print(f"  Inliers: {inliers}/{num_points_used}")

    print(f"  Avg error:    {avg_error:.2f}mm")
    print(f"  Median error: {median_error:.2f}mm")
    print(f"  Max error:    {max_error:.2f}mm")

    # Per-zone error analysis
    zone_errors_mm = [[[] for _ in range(ZONE_GRID[1])] for _ in range(ZONE_GRID[0])]
    rb_x_range = (all_rb[:, 0].min(), all_rb[:, 0].max())
    rb_y_range = (all_rb[:, 1].min(), all_rb[:, 1].max())
    x_span = max(rb_x_range[1] - rb_x_range[0], 1)
    y_span = max(rb_y_range[1] - rb_y_range[0], 1)

    print(f"\n  ── Error by zone (robot space) ──")
    for i in range(len(all_rb)):
        # Map robot coords to zone grid
        nx = (all_rb[i][0] - rb_x_range[0]) / x_span
        ny = (all_rb[i][1] - rb_y_range[0]) / y_span
        zc = min(int(nx * ZONE_GRID[1]), ZONE_GRID[1] - 1)
        zr = min(int(ny * ZONE_GRID[0]), ZONE_GRID[0] - 1)
        zone_errors_mm[zr][zc].append(errors[i])

    zone_labels_r = ["Near", "Mid", "Far"]
    zone_labels_c = ["Left", "Center", "Right"]
    for r in range(ZONE_GRID[0]):
        for c in range(ZONE_GRID[1]):
            errs = zone_errors_mm[r][c]
            label = f"{zone_labels_r[r]}-{zone_labels_c[c]}"
            if errs:
                mean_e = np.mean(errs)
                max_e = np.max(errs)
                status = "✓" if mean_e < 1.0 else "~" if mean_e < 2.0 else "✗"
                print(f"    {status} {label}: avg={mean_e:.2f}mm max={max_e:.2f}mm ({len(errs)}pts)")
            else:
                print(f"    - {label}: NO DATA (move board here!)")

    if avg_error < 1.0:
        print(f"\n  ✓ Excellent calibration!")
    elif avg_error < 2.0:
        print(f"\n  ~ Good — try more captures at edges for better coverage")
    else:
        print(f"\n  ✗ High error — check axis mapping (F) or add more captures")

    return True


def px_to_robot(px, py):
    if homography is None:
        return None
    h = homography @ [px, py, 1.0]
    return h[0] / h[2], h[1] / h[2]


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


def on_mouse(event, x, y, flags, param):
    global last_click
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    last_click = (x, y)

    if mode == "pick":
        result = px_to_robot(x, y)
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


# ─── Load camera calibration + undistortion maps ───
cam_matrix = None
dist_coeffs = None

if os.path.exists(npz_path):
    npz = np.load(npz_path)
    if "map1" in npz and "map2" in npz:
        remap_map1 = npz["map1"]
        remap_map2 = npz["map2"]
        cam_matrix = npz["camera_matrix"]
        dist_coeffs = npz["dist_coeffs"]
        if "new_camera_matrix" in npz:
            undist_cam_matrix = npz["new_camera_matrix"]
        else:
            undist_cam_matrix = cam_matrix
        use_undistort = True
        print(f"✓ Loaded undistortion maps from {npz_path}")
    else:
        cam_matrix = npz["camera_matrix"]
        dist_coeffs = npz["dist_coeffs"]
        print(f"✓ Loaded camera calibration (no precomputed maps — will undistort on the fly)")

if not use_undistort and os.path.exists(calib_path):
    with open(calib_path) as f:
        calib = json.load(f)
    cam_matrix = np.array(calib["camera_matrix"])
    dist_coeffs = np.array(calib["dist_coeffs"])
    print(f"✓ Loaded camera_calibration.json (reproj err: {calib.get('reprojection_error', '?')})")

    if "new_camera_matrix" in calib:
        undist_cam_matrix = np.array(calib["new_camera_matrix"])
    else:
        undist_cam_matrix = cam_matrix

if cam_matrix is not None and not use_undistort:
    # Compute maps on the fly if not precomputed
    print("  Computing undistortion maps...")

# ─── Setup ChArUco ───
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard(BOARD_SQUARES, SQUARE_SIZE / 1000, MARKER_SIZE / 1000, dictionary)
charuco_detector = cv2.aruco.CharucoDetector(board)

# ─── Load saved calibration ───
if os.path.exists(transform_path):
    with open(transform_path) as f:
        saved = json.load(f)
    homography = np.array(saved["homography"])
    board_offset_x = saved.get("board_offset_x", 85.0)
    board_offset_y = saved.get("board_offset_y", 0.0)
    axis_mapping = saved.get("axis_mapping", 0)
    PICK_Z = saved.get("pick_z", 100)
    avg_error = saved.get("avg_error")
    num_points_used = saved.get("num_points", 0)
    capture_count = saved.get("num_captures", 0)
    mode = "pick"
    print(f"✓ Loaded saved calibration ({num_points_used} pts, err={avg_error:.2f}mm)")
    print(f"  → PICK mode ready!")

# ─── Open camera ───
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("✗ Cannot open camera!")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

w = int(cap.get(3))
h = int(cap.get(4))

# Compute undistortion maps if we have calibration but no precomputed maps
if cam_matrix is not None and not use_undistort:
    if undist_cam_matrix is None:
        undist_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(
            cam_matrix, dist_coeffs, (w, h), alpha=0
        )
    remap_map1, remap_map2 = cv2.initUndistortRectifyMap(
        cam_matrix, dist_coeffs, None, undist_cam_matrix, (w, h), cv2.CV_32FC1
    )
    use_undistort = True
    print(f"  ✓ Undistortion maps computed for {w}x{h}")

print()
print("═" * 60)
print("  IRIS Vision v2 — ChArUco Calibration + Pick")
print("═" * 60)
print(f"  Camera: {w}x{h}")
undist_str = "ON ✓" if use_undistort else "OFF ⚠ (run calibrate_camera.py first!)"
print(f"  Undistortion: {undist_str}")
print(f"  Board:  {BOARD_SQUARES[0]}x{BOARD_SQUARES[1]}, square={SQUARE_SIZE}mm")
print(f"  Offset: X={board_offset_x}mm  Y={board_offset_y}mm")
print()
print("  Workflow:")
print("    1. Show board at position A → SPACE (capture)")
print("    2. Move board to position B → SPACE (capture)")
print("    3. Repeat 3-5 positions covering workspace")
print("    4. Press C to compute homography from ALL captures")
print()
print("  Keys: SPACE=capture C=calibrate F=flip O=offset")
print("        T=test G=grid E=errors S=save +/-=Z R=redo Q=quit")
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

    # ─── UNDISTORT ─── (the key fix for edge accuracy!)
    if use_undistort:
        frame = cv2.remap(frame, remap_map1, remap_map2, cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display = frame.copy()

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

        # Draw pose axes if camera is calibrated
        # Use undistorted camera matrix if available
        draw_cam = undist_cam_matrix if use_undistort else cam_matrix
        if draw_cam is not None:
            obj_points_3d = board.getChessboardCorners()[charuco_ids.flatten()]
            # After undistortion, distortion coeffs are zero
            draw_dist = np.zeros(5) if use_undistort else dist_coeffs
            success, rvec, tvec = cv2.solvePnP(obj_points_3d, charuco_corners, draw_cam, draw_dist)
            if success:
                cv2.drawFrameAxes(display, draw_cam, draw_dist, rvec, tvec, 0.05)

        # Show axis mapping preview
        cfg = AXIS_CONFIGS[axis_mapping]
        robot_pts = get_board_corners_robot(charuco_ids, cfg)
        for i in range(0, len(charuco_corners), max(1, len(charuco_corners) // 6)):
            px = charuco_corners[i].flatten()
            rp = robot_pts[i]
            cv2.putText(display, f"({rp[0]:.0f},{rp[1]:.0f})",
                        (int(px[0]) + 8, int(px[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    # ─── Error map overlay ───
    if show_error_map and zone_errors_mm is not None and homography is not None:
        try:
            h_inv = np.linalg.inv(homography)
            rb_x_range = (WORKSPACE_X_MIN, WORKSPACE_X_MAX)
            rb_y_range = (WORKSPACE_Y_MIN, WORKSPACE_Y_MAX)
            x_step = (rb_x_range[1] - rb_x_range[0]) / ZONE_GRID[1]
            y_step = (rb_y_range[1] - rb_y_range[0]) / ZONE_GRID[0]

            for r in range(ZONE_GRID[0]):
                for c in range(ZONE_GRID[1]):
                    errs = zone_errors_mm[r][c]
                    if not errs:
                        continue

                    mean_e = np.mean(errs)
                    # Map zone center back to pixel coords
                    rx = rb_x_range[0] + (c + 0.5) * x_step
                    ry = rb_y_range[0] + (r + 0.5) * y_step
                    p = h_inv @ [rx, ry, 1.0]
                    px_x, px_y = int(p[0] / p[2]), int(p[1] / p[2])

                    if 0 <= px_x < w and 0 <= px_y < h:
                        if mean_e < 1.0:
                            color = (0, 255, 0)
                        elif mean_e < 2.0:
                            color = (0, 200, 255)
                        else:
                            color = (0, 0, 255)

                        cv2.circle(display, (px_x, px_y), 25, color, 3)
                        cv2.putText(display, f"{mean_e:.1f}mm",
                                    (px_x - 25, px_y - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except:
            pass

    # ─── Grid overlay ───
    if show_grid and homography is not None:
        try:
            h_inv = np.linalg.inv(homography)
            grid_step = 50

            for rx in range(WORKSPACE_X_MIN, WORKSPACE_X_MAX + 1, grid_step):
                pts_line = []
                for ry in range(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX + 1, 10):
                    p = h_inv @ [rx, ry, 1.0]
                    px_x, px_y = int(p[0] / p[2]), int(p[1] / p[2])
                    if 0 <= px_x < w and 0 <= px_y < h:
                        pts_line.append((px_x, px_y))
                for j in range(len(pts_line) - 1):
                    cv2.line(display, pts_line[j], pts_line[j + 1], (50, 50, 50), 1)

            for ry in range(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX + 1, grid_step):
                pts_line = []
                for rx in range(WORKSPACE_X_MIN, WORKSPACE_X_MAX + 1, 10):
                    p = h_inv @ [rx, ry, 1.0]
                    px_x, px_y = int(p[0] / p[2]), int(p[1] / p[2])
                    if 0 <= px_x < w and 0 <= px_y < h:
                        pts_line.append((px_x, px_y))
                for j in range(len(pts_line) - 1):
                    cv2.line(display, pts_line[j], pts_line[j + 1], (50, 50, 50), 1)

            corners_robot = [
                [WORKSPACE_X_MIN, WORKSPACE_Y_MIN],
                [WORKSPACE_X_MAX, WORKSPACE_Y_MIN],
                [WORKSPACE_X_MAX, WORKSPACE_Y_MAX],
                [WORKSPACE_X_MIN, WORKSPACE_Y_MAX],
            ]
            corners_px = []
            for cx, cy in corners_robot:
                p = h_inv @ [cx, cy, 1.0]
                corners_px.append((int(p[0] / p[2]), int(p[1] / p[2])))
            for j in range(4):
                pt1 = corners_px[j]
                pt2 = corners_px[(j + 1) % 4]
                if all(0 <= c < max(w, h) for c in pt1 + pt2):
                    cv2.line(display, pt1, pt2, (0, 100, 255), 2)
        except:
            pass

    # ─── Last click ───
    if last_click and mode == "pick":
        cv2.circle(display, last_click, 12, (0, 255, 255), 2)
        cv2.drawMarker(display, last_click, (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        result = px_to_robot(last_click[0], last_click[1])
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
            txt = f"BOARD | {num_corners}pts | captures={capture_count} | mapping={axis_mapping} | SPACE=add C=calibrate"
            col = (0, 255, 0)
        else:
            txt = f"Show board... | captures={capture_count}"
            col = (0, 165, 255)
    else:
        txt = f"PICK | Z={PICK_Z}mm"
        if test_mode:
            txt += " | TEST"
        if avg_error is not None:
            txt += f" | err={avg_error:.1f}mm"
        txt += f" | {num_points_used}pts/{capture_count}cap"
        col = (0, 255, 0) if not test_mode else (255, 200, 0)

    cv2.putText(display, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2)

    # Second line
    cfg = AXIS_CONFIGS[axis_mapping]
    undist_label = "UNDIST" if use_undistort else "RAW⚠"
    cv2.putText(display, f"{undist_label} | Offset: X={board_offset_x:.0f} Y={board_offset_y:.0f}mm | {cfg['label']}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # ─── Footer ───
    cv2.rectangle(display, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.putText(display, "SPACE=add C=cal F=flip O=off T=test G=grid E=err S=save +/-=Z R=redo Q=quit",
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
                    print(f"  ✓ Offset set: X={board_offset_x}mm Y={board_offset_y}mm")
                    print(f"    Captures cleared — recapture with new offset, then press C")
                    all_px_points.clear()
                    all_robot_points.clear()
                    capture_count = 0
                    waiting_for_offset = False
                except ValueError:
                    print("  ✗ Type: X Y  (e.g. 85 0)")
    except queue.Empty:
        pass

    # ─── Key handling ───
    if key == ord('q') or key == 27:
        break

    elif key == ord(' '):
        # Add current detection to buffer
        if last_charuco_corners is not None and last_charuco_ids is not None:
            add_capture(last_charuco_corners, last_charuco_ids)
            # Flash
            flash = display.copy()
            cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), 6)
            cv2.imshow("IRIS Vision", flash)
            cv2.waitKey(100)
        else:
            print("  ✗ No board detected — show it first")

    elif key == ord('c'):
        # Compute homography from all captures
        if compute_calibration_multi():
            mode = "pick"
            print("  → PICK mode! Press T for test, click to move robot.")
        else:
            print("  ✗ Calibration failed — add more captures with SPACE")

    elif key == ord('f'):
        axis_mapping = (axis_mapping + 1) % 4
        cfg = AXIS_CONFIGS[axis_mapping]
        print(f"  Axis mapping {axis_mapping}: {cfg['label']}")
        # Recompute robot points for existing captures
        if capture_count > 0:
            # Need to re-derive robot points from stored pixel data
            # Easiest: clear and recapture
            print(f"  ⚠ Axis changed — captures cleared, recapture with SPACE then C")
            all_px_points.clear()
            all_robot_points.clear()
            capture_count = 0
            homography = None
            mode = "detect"

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

    elif key == ord('e'):
        show_error_map = not show_error_map
        print(f"  Error map {'ON' if show_error_map else 'OFF'}")

    elif key == ord('s'):
        if homography is not None:
            save_data = {
                "homography": homography.tolist(),
                "board_offset_x": board_offset_x,
                "board_offset_y": board_offset_y,
                "axis_mapping": axis_mapping,
                "axis_label": AXIS_CONFIGS[axis_mapping]["label"],
                "pick_z": PICK_Z,
                "avg_error": float(avg_error) if avg_error else None,
                "num_points": num_points_used,
                "num_captures": capture_count,
                "undistorted": use_undistort,
                "board_squares": list(BOARD_SQUARES),
                "square_size_mm": SQUARE_SIZE,
            }
            with open(transform_path, "w") as f:
                json.dump(save_data, f, indent=2)
            print(f"  ✓ Saved to {transform_path}")
        else:
            print("  ✗ Nothing to save — calibrate first")

    elif key == ord('r'):
        homography = None
        mode = "detect"
        last_click = None
        avg_error = None
        zone_errors_mm = None
        all_px_points.clear()
        all_robot_points.clear()
        capture_count = 0
        print("  Reset — show board, SPACE to capture, C to calibrate")

    elif key == ord('+') or key == ord('='):
        PICK_Z += 10
        print(f"  Z = {PICK_Z}mm")

    elif key == ord('-'):
        PICK_Z = max(10, PICK_Z - 10)
        print(f"  Z = {PICK_Z}mm")

cap.release()
cv2.destroyAllWindows()
