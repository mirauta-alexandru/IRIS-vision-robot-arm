#!/usr/bin/env python3
"""
IRIS Vision Pick — Guided calibration with coverage validation

1. Move robot to suggested positions (or your own)
2. Click gripper tip in camera
3. Type X Y from visualizer
4. Script validates coverage & accuracy before enabling pick mode

Keys:
  C = calibrate (compute homography)
  S = save points
  U = undo last point
  R = reset all points
  +/- = adjust Z
  G = toggle grid overlay
  T = test mode (click to see predicted coords without moving)
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
points_path = os.path.join(SAVE_DIR, "vision_points.json")

# Workspace bounds (mm) — adjust to your IRIS arm reach
WORKSPACE_X_MIN = 80
WORKSPACE_X_MAX = 350
WORKSPACE_Y_MIN = -120
WORKSPACE_Y_MAX = 120

# Calibration thresholds
MIN_POINTS = 6
MAX_ALLOWED_ERROR = 5.0  # mm — warn above
TARGET_ERROR = 3.0       # mm — good

# ─── State ───
cal_points = []
homography = None
mode = "calibrate"
last_click = None
pending_click = None
input_q = queue.Queue()
show_grid = True
test_mode = False
point_errors = []


def workspace_coverage(points):
    """Check how well points cover the workspace. Returns score 0-100 and issues."""
    if len(points) < 4:
        return 0, ["Need at least 4 points"]

    robot_pts = np.array([[p[2], p[3]] for p in points])
    x_coords = robot_pts[:, 0]
    y_coords = robot_pts[:, 1]

    issues = []
    score = 100

    # Check X spread
    x_range = x_coords.max() - x_coords.min()
    x_workspace = WORKSPACE_X_MAX - WORKSPACE_X_MIN
    x_coverage = x_range / x_workspace * 100
    if x_coverage < 50:
        issues.append(f"X spread too small: {x_range:.0f}mm ({x_coverage:.0f}% of workspace)")
        score -= 30

    # Check Y spread
    y_range = y_coords.max() - y_coords.min()
    y_workspace = WORKSPACE_Y_MAX - WORKSPACE_Y_MIN
    y_coverage = y_range / y_workspace * 100
    if y_coverage < 50:
        issues.append(f"Y spread too small: {y_range:.0f}mm ({y_coverage:.0f}% of workspace)")
        score -= 30

    # Check for collinearity
    if len(points) >= 3:
        centered = robot_pts - robot_pts.mean(axis=0)
        _, s, _ = np.linalg.svd(centered)
        if len(s) >= 2 and s[1] > 0:
            ratio = s[0] / s[1]
            if ratio > 5:
                issues.append("Points nearly collinear! Spread Y more.")
                score -= 30

    # Check pixel spread
    px_pts = np.array([[p[0], p[1]] for p in points])
    px_x_range = px_pts[:, 0].max() - px_pts[:, 0].min()
    px_y_range = px_pts[:, 1].max() - px_pts[:, 1].min()
    if px_x_range < 400:
        issues.append(f"Pixel X range only {px_x_range:.0f}px — click across wider area")
        score -= 15
    if px_y_range < 300:
        issues.append(f"Pixel Y range only {px_y_range:.0f}px — click across taller area")
        score -= 15

    # Check quadrant coverage
    x_mid = (x_coords.max() + x_coords.min()) / 2
    y_mid = (y_coords.max() + y_coords.min()) / 2
    quadrants = set()
    for x, y in zip(x_coords, y_coords):
        qx = "R" if x >= x_mid else "L"
        qy = "T" if y >= y_mid else "B"
        quadrants.add(qx + qy)
    if len(quadrants) < 3:
        issues.append(f"Only {len(quadrants)}/4 quadrants covered — spread points more")
        score -= 20

    return max(0, score), issues


def compute_homography():
    global homography, point_errors
    if len(cal_points) < 4:
        print(f"  ✗ Need 4+ points (have {len(cal_points)})")
        return False

    src = np.float64([[p[0], p[1]] for p in cal_points])
    dst = np.float64([[p[2], p[3]] for p in cal_points])
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if homography is None:
        print("  ✗ Homography computation failed!")
        return False

    # Per-point errors
    point_errors = []
    outliers = []
    for i, p in enumerate(cal_points):
        h = homography @ [p[0], p[1], 1]
        xy = h[:2] / h[2]
        err = np.sqrt((xy[0] - p[2]) ** 2 + (xy[1] - p[3]) ** 2)
        point_errors.append(err)
        if err > MAX_ALLOWED_ERROR * 2:
            outliers.append(i)

    avg_err = np.mean(point_errors)
    max_err = np.max(point_errors)

    print()
    print(f"  ── Calibration Results ──")
    print(f"  Points:    {len(cal_points)}")
    print(f"  Avg error: {avg_err:.1f}mm {'✓' if avg_err <= TARGET_ERROR else '⚠' if avg_err <= MAX_ALLOWED_ERROR else '✗'}")
    print(f"  Max error: {max_err:.1f}mm {'✓' if max_err <= MAX_ALLOWED_ERROR else '✗'}")

    if outliers:
        print(f"  ⚠ Outliers (err > {MAX_ALLOWED_ERROR*2:.0f}mm): {['P'+str(i+1) for i in outliers]}")
        print(f"    → Remove them with U and re-add carefully")

    for i, (p, e) in enumerate(zip(cal_points, point_errors)):
        marker = "✓" if e <= TARGET_ERROR else "⚠" if e <= MAX_ALLOWED_ERROR else "✗"
        print(f"    P{i+1}: px({p[0]},{p[1]}) → ({p[2]:.0f},{p[3]:.0f})mm  err={e:.1f}mm {marker}")

    # Coverage check
    coverage_score, issues = workspace_coverage(cal_points)
    if issues:
        print(f"  ── Coverage ({coverage_score}%) ──")
        for issue in issues:
            print(f"    ⚠ {issue}")
    else:
        print(f"  ── Coverage: {coverage_score}% ✓ ──")

    if avg_err <= TARGET_ERROR and coverage_score >= 60:
        print(f"  ✓ Good calibration!")
    elif avg_err <= MAX_ALLOWED_ERROR:
        print(f"  ~ Acceptable, could improve")
    else:
        print(f"  ✗ Poor — fix outlier points")

    return True


def px_to_robot(px, py):
    if homography is None:
        return None
    h = homography @ [px, py, 1.0]
    return h[0] / h[2], h[1] / h[2]


def is_in_workspace(rx, ry):
    margin = 50  # mm tolerance
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
    global last_click, pending_click
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    last_click = (x, y)

    if mode == "calibrate":
        pending_click = (x, y)
        print(f"\n  Clicked ({x}, {y}) — type X Y in terminal:")

    elif mode == "pick":
        result = px_to_robot(x, y)
        if result:
            rx, ry = result
            if not is_in_workspace(rx, ry):
                print(f"\n  ✗ ({rx:.0f}, {ry:.0f})mm OUTSIDE workspace — don't trust this area")
                return
            if test_mode:
                print(f"\n  [TEST] ({x},{y}) → ({rx:.1f}, {ry:.1f}, {PICK_Z})mm")
            else:
                print(f"\n  Click ({x},{y}) → Robot ({rx:.1f}, {ry:.1f}, {PICK_Z})mm")
                err = send_ik(rx, ry, PICK_Z)
                print(f"    → {'OK err=' + str(round(err, 1)) + 'mm' if err >= 0 else 'FAILED (bridge?)'}")


def stdin_reader():
    while True:
        try:
            input_q.put(input().strip())
        except:
            break


def suggest_positions():
    return [
        (WORKSPACE_X_MIN + 30, 0, "center-close"),
        (WORKSPACE_X_MAX - 30, 0, "center-far"),
        ((WORKSPACE_X_MIN + WORKSPACE_X_MAX) // 2, WORKSPACE_Y_MIN + 20, "mid-left"),
        ((WORKSPACE_X_MIN + WORKSPACE_X_MAX) // 2, WORKSPACE_Y_MAX - 20, "mid-right"),
        (WORKSPACE_X_MIN + 40, WORKSPACE_Y_MIN + 30, "near-left"),
        (WORKSPACE_X_MIN + 40, WORKSPACE_Y_MAX - 30, "near-right"),
        (WORKSPACE_X_MAX - 40, WORKSPACE_Y_MIN + 30, "far-left"),
        (WORKSPACE_X_MAX - 40, WORKSPACE_Y_MAX - 30, "far-right"),
    ]


# ─── Start ───
print()
print("═" * 55)
print("  IRIS Vision Pick — Guided Calibration")
print("═" * 55)
print()
print("  Suggested positions (move via visualizer):")
for i, (x, y, name) in enumerate(suggest_positions()):
    print(f"    {i+1}. X={x:>4} Y={y:>4}  ({name})")
print()
print("  Steps:")
print("    1. Move robot to a position")
print("    2. Click gripper tip in camera")
print("    3. Type exact X Y from visualizer")
print("    4. Repeat 6-8 times, spread across workspace")
print("    5. Press C to calibrate")
print()
print("  Keys: C=cal S=save U=undo R=reset G=grid T=test +/-=Z Q=quit")
print("═" * 55)

# Load saved points
if os.path.exists(points_path):
    with open(points_path) as f:
        saved = json.load(f)
    cal_points = saved.get("points", [])
    PICK_Z = saved.get("pick_z", 100)
    print(f"  ✓ Loaded {len(cal_points)} saved points")

    coverage_score, issues = workspace_coverage(cal_points)
    if issues:
        print(f"  Coverage issues:")
        for issue in issues:
            print(f"    ⚠ {issue}")

    if len(cal_points) >= 4:
        if compute_homography():
            mode = "pick"
            print("  → PICK mode ready! Press T for test mode first.")

cap = cv2.VideoCapture(CAMERA_ID)
w = int(cap.get(3))
h = int(cap.get(4))
print(f"  ✓ Camera {CAMERA_ID}: {w}x{h}")

cv2.namedWindow("IRIS Vision")
cv2.setMouseCallback("IRIS Vision", on_mouse)
threading.Thread(target=stdin_reader, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    display = frame.copy()

    # ─── Grid overlay ───
    if show_grid and homography is not None:
        h_inv = np.linalg.inv(homography)
        grid_step = 50  # mm

        # Vertical lines (constant X)
        for rx in range(WORKSPACE_X_MIN, WORKSPACE_X_MAX + 1, grid_step):
            pts_line = []
            for ry in range(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX + 1, 10):
                p = h_inv @ [rx, ry, 1.0]
                px_x, px_y = int(p[0] / p[2]), int(p[1] / p[2])
                if 0 <= px_x < w and 0 <= px_y < h:
                    pts_line.append((px_x, px_y))
            for j in range(len(pts_line) - 1):
                cv2.line(display, pts_line[j], pts_line[j + 1], (50, 50, 50), 1)

        # Horizontal lines (constant Y)
        for ry in range(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX + 1, grid_step):
            pts_line = []
            for rx in range(WORKSPACE_X_MIN, WORKSPACE_X_MAX + 1, 10):
                p = h_inv @ [rx, ry, 1.0]
                px_x, px_y = int(p[0] / p[2]), int(p[1] / p[2])
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
            p = h_inv @ [cx, cy, 1.0]
            corners_px.append((int(p[0] / p[2]), int(p[1] / p[2])))
        for j in range(4):
            cv2.line(display, corners_px[j], corners_px[(j + 1) % 4], (0, 100, 255), 2)

    # ─── Calibration points ───
    for i, p in enumerate(cal_points):
        pt = (int(p[0]), int(p[1]))
        if point_errors and i < len(point_errors):
            err = point_errors[i]
            color = (0, 255, 0) if err <= TARGET_ERROR else (0, 200, 255) if err <= MAX_ALLOWED_ERROR else (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.circle(display, pt, 8, color, 2)
        cv2.putText(display, f"P{i+1}({p[2]:.0f},{p[3]:.0f})",
                    (pt[0] + 12, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if point_errors and i < len(point_errors):
            cv2.putText(display, f"{point_errors[i]:.1f}mm",
                        (pt[0] + 12, pt[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # ─── Last click ───
    if last_click:
        cv2.circle(display, last_click, 12, (0, 255, 255), 2)
        cv2.drawMarker(display, last_click, (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        if mode == "pick":
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
    cv2.rectangle(display, (0, 0), (w, 50), (20, 20, 20), -1)
    if mode == "calibrate":
        txt = f"CALIBRATE | {len(cal_points)} pts"
        coverage_score, _ = workspace_coverage(cal_points)
        txt += f" | coverage: {coverage_score}%"
        if len(cal_points) < MIN_POINTS:
            txt += f" | need {MIN_POINTS - len(cal_points)} more"
        col = (0, 165, 255)
    else:
        txt = f"PICK | Z={PICK_Z}mm"
        if test_mode:
            txt += " | TEST (no move)"
        if point_errors:
            txt += f" | avg err: {np.mean(point_errors):.1f}mm"
        col = (0, 255, 0) if not test_mode else (255, 200, 0)
    cv2.putText(display, txt, (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # ─── Footer ───
    cv2.rectangle(display, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.putText(display, "C=cal S=save U=undo R=reset G=grid T=test +/-=Z Q=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("IRIS Vision", display)
    key = cv2.waitKey(30) & 0xFF

    # ─── Terminal input ───
    try:
        line = input_q.get_nowait()
        if pending_click and mode == "calibrate":
            parts = line.split()
            if len(parts) >= 2:
                try:
                    rx, ry = float(parts[0]), float(parts[1])
                    rz = float(parts[2]) if len(parts) >= 3 else PICK_Z
                    cal_points.append([pending_click[0], pending_click[1], rx, ry, rz])
                    print(f"  ✓ P{len(cal_points)}: px({pending_click[0]},{pending_click[1]}) → ({rx},{ry},{rz})mm")

                    coverage_score, issues = workspace_coverage(cal_points)
                    if issues and len(cal_points) >= 3:
                        for issue in issues[:2]:
                            print(f"    hint: {issue}")
                    elif coverage_score >= 80 and len(cal_points) >= MIN_POINTS:
                        print(f"    ✓ Good coverage ({coverage_score}%)! Press C to calibrate.")

                    pending_click = None
                except ValueError:
                    print("  ✗ Invalid — type: X Y  (e.g. 200 -50)")
            else:
                print("  ✗ Type: X Y  (e.g. 200 -50)")
    except queue.Empty:
        pass

    # ─── Keys ───
    if key == ord('q') or key == 27:
        break

    elif key == ord('c'):
        if compute_homography():
            mode = "pick"
            print("  → PICK mode! Press T for test mode first.")

    elif key == ord('s'):
        with open(points_path, "w") as f:
            json.dump({"points": cal_points, "pick_z": PICK_Z,
                       "workspace": {"x_min": WORKSPACE_X_MIN, "x_max": WORKSPACE_X_MAX,
                                     "y_min": WORKSPACE_Y_MIN, "y_max": WORKSPACE_Y_MAX}}, f, indent=2)
        print(f"  ✓ Saved {len(cal_points)} points")

    elif key == ord('u') and cal_points:
        removed = cal_points.pop()
        point_errors.clear()
        print(f"  Undo — {len(cal_points)} remaining")
        if homography is not None and len(cal_points) >= 4:
            compute_homography()

    elif key == ord('r'):
        cal_points.clear()
        point_errors.clear()
        homography = None
        mode = "calibrate"
        last_click = None
        print("  Reset — all points cleared")

    elif key == ord('g'):
        show_grid = not show_grid
        print(f"  Grid {'ON' if show_grid else 'OFF'}")

    elif key == ord('t'):
        test_mode = not test_mode
        print(f"  Test mode {'ON — click shows coords only' if test_mode else 'OFF — click moves robot'}")

    elif key == ord('+') or key == ord('='):
        PICK_Z += 10
        print(f"  Z = {PICK_Z}mm")

    elif key == ord('-'):
        PICK_Z = max(10, PICK_Z - 10)
        print(f"  Z = {PICK_Z}mm")

cap.release()
cv2.destroyAllWindows()
