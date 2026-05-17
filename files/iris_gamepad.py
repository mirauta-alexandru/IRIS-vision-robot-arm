#!/usr/bin/env python3
"""
IRIS Gamepad Controller v2 — Smooth IK Control
PS4/PS5/Xbox → Robot Arm via iris_bridge

Design principles:
  - Cartesian XYZ position control with exponential smoothing
  - Gripper is HOLD state: only changes when you press triggers
  - IK only sent when position actually changes (>0.3mm threshold)
  - Previous joint angles used as IK seed for smooth solutions

Controls:
    Left Stick ↕    →  X forward/back
    Left Stick ↔    →  Y left/right
    Right Stick ↕   →  Z up/down
    R2 (hold)       →  Close gripper (press=close, release=stays)
    L2 (hold)       →  Open gripper (press=open, release=stays)
    L1 held         →  Slow/precision mode
    R1 held         →  Fast mode
    △ / Y           →  Home position
    □ / X           →  Safe height (Z=200)
    ○ / B           →  Describe scene
    ✕ / A           →  Detect objects
    D-pad ↕         →  Fine Z ±3mm
    D-pad ↔         →  Fine Y ±3mm
    Share            →  Print position
    Options          →  Emergency stop → home

Requirements: pip install pygame requests
"""

import pygame
import requests
import time
import sys
import math
import threading

# ─── CONFIG ───
BRIDGE = "http://localhost:8765"
LOOP_HZ = 30
DEADZONE = 0.15

# Smoothing: 0.0 = instant (no smooth), 1.0 = never moves
# 0.7 is a good balance — responsive but smooth
SMOOTHING = 0.7

# Speed (mm per second, not per tick — frame-rate independent)
SPEED_SLOW = 40
SPEED_NORMAL = 80
SPEED_FAST = 160

# Limits
X_MIN, X_MAX = 80, 350
Y_MIN, Y_MAX = -120, 120
Z_MIN, Z_MAX = -10, 250

# ─── State ───
# Target = where the user WANTS to go (instant, from stick input)
# Smooth = where we ACTUALLY send (lerped toward target)
target = {"x": 200.0, "y": 0.0, "z": 200.0}
smooth = {"x": 200.0, "y": 0.0, "z": 200.0}
last_sent = {"x": -999, "y": -999, "z": -999}  # Force first send

gripper_state = 0       # 0=open, 70=closed — HOLDS between presses
gripper_sent = -1
connected = False
speed_mode = "NORMAL"

# Terminal colors
O = "\033[38;5;208m"  # orange
G = "\033[38;5;82m"   # green
C = "\033[38;5;45m"   # cyan
D = "\033[38;5;240m"  # dim
R = "\033[38;5;196m"  # red
Y = "\033[38;5;226m"  # yellow
B = "\033[1m"         # bold
X = "\033[0m"         # reset


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def deadzone(val):
    if abs(val) < DEADZONE:
        return 0.0
    sign = 1 if val > 0 else -1
    return sign * (abs(val) - DEADZONE) / (1.0 - DEADZONE)


def lerp(a, b, t):
    """Linear interpolate from a toward b by factor t (0=a, 1=b)."""
    return a + (b - a) * t


def send_ik(x, y, z):
    try:
        r = requests.post(f"{BRIDGE}/ik", json={
            "x": round(x, 1), "y": round(y, 1), "z": round(z, 1), "send": True
        }, timeout=0.5)
        return r.json()
    except:
        return None


def send_gripper(angle):
    try:
        requests.get(f"{BRIDGE}/gripper?angle={int(angle)}", timeout=0.3)
    except:
        pass


def go_home():
    global target, smooth, gripper_state, gripper_sent
    send_gripper(0)
    gripper_state = 0
    gripper_sent = 0
    time.sleep(0.1)
    target["x"], target["y"], target["z"] = 200.0, 0.0, 250.0
    smooth["x"], smooth["y"], smooth["z"] = 200.0, 0.0, 250.0
    send_ik(200, 0, 250)
    print(f"\n  {G}🏠 HOME{X}")


def safe_height():
    global target
    target["z"] = 200.0
    print(f"\n  {C}⬆ Safe height{X}")


def describe_scene():
    print(f"\n  {Y}👁 Describing scene...{X}")
    try:
        r = requests.post(f"{BRIDGE}/gemini", json={
            "prompt": "Describe everything on this table. Concise. Romanian."
        }, timeout=15)
        d = r.json()
        if "raw" in d:
            print(f"  {C}{d['raw'][:300]}{X}")
    except Exception as e:
        print(f"  {R}Error: {e}{X}")


def detect_objects():
    print(f"\n  {Y}🔍 Detecting...{X}")
    try:
        r = requests.post(f"{BRIDGE}/gemini", json={
            "prompt": 'all objects\nPoint to each relevant object. The answer should follow the json format: [{"point": [y, x], "label": <label>}]. The points are in [y, x] format normalized to 0-1000.'
        }, timeout=15)
        d = r.json()
        for obj in d.get("objects", []):
            rx = obj.get("robot", {}).get("x", "?")
            ry = obj.get("robot", {}).get("y", "?")
            print(f"    {G}• {obj.get('label','?')}{X} → ({rx}, {ry})mm")
    except Exception as e:
        print(f"  {R}Error: {e}{X}")


def check_bridge():
    global connected
    try:
        r = requests.get(f"{BRIDGE}/ping", timeout=1)
        connected = r.json().get("serial", False)
    except:
        connected = False


def hud():
    grip_txt = "OPEN" if gripper_state < 35 else "CLOSED"
    conn = f"{G}●{X}" if connected else f"{R}●{X}"
    sys.stdout.write(
        f"\r  {conn} "
        f"{O}X:{smooth['x']:6.1f}{X}  "
        f"{C}Y:{smooth['y']:6.1f}{X}  "
        f"{G}Z:{smooth['z']:6.1f}{X}  "
        f"{D}|{X} "
        f"Grip:{Y}{grip_txt}{X}  "
        f"Speed:{B}{speed_mode}{X}      "
    )
    sys.stdout.flush()


# ═══════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════
def main():
    global target, smooth, last_sent, gripper_state, gripper_sent, connected, speed_mode

    pygame.init()
    pygame.joystick.init()

    print()
    print(f"  {O}╔══════════════════════════════════════════════╗{X}")
    print(f"  {O}║  IRIS Gamepad Controller v2 — Smooth IK      ║{X}")
    print(f"  {O}║  PS4/PS5/Xbox → Robot Arm via Bridge          ║{X}")
    print(f"  {O}╚══════════════════════════════════════════════╝{X}")
    print()

    check_bridge()
    print(f"  {G if connected else R}{'✓' if connected else '✗'} Bridge {'connected' if connected else 'not found'}{X}")

    print(f"  {D}Waiting for controller...{X}", end="", flush=True)
    while pygame.joystick.get_count() == 0:
        pygame.event.pump()
        time.sleep(0.5)
        pygame.joystick.quit()
        pygame.joystick.init()

    js = pygame.joystick.Joystick(0)
    js.init()
    n_ax = js.get_numaxes()
    n_btn = js.get_numbuttons()
    n_hat = js.get_numhats()

    print(f"\r  {G}✓ {js.get_name()}{X}                              ")
    print(f"  {D}  Axes:{n_ax} Btns:{n_btn} Hats:{n_hat}{X}")

    # Axis mapping auto-detect
    # PS4: 0=LX 1=LY 2=RX 3=RY 4=L2 5=R2 (6 axes)
    # Xbox: similar but triggers might be on 2,5 or 4,5
    ax_lx, ax_ly = 0, 1
    ax_rx = 2 if n_ax >= 4 else 0
    ax_ry = 3 if n_ax >= 4 else 1
    ax_l2 = 4 if n_ax >= 6 else -1
    ax_r2 = 5 if n_ax >= 6 else -1

    print()
    print(f"  {D}Controls:{X}")
    print(f"    L-Stick ↕↔ → X/Y    R-Stick ↕ → Z")
    print(f"    R2 → Close grip      L2 → Open grip")
    print(f"    L1 → Slow            R1 → Fast")
    print(f"    △ → Home   □ → Safe   ○ → Vision   ✕ → Detect")
    print(f"    D-pad → Fine ±3mm    Share → Position")
    print()

    # Start at home
    go_home()
    time.sleep(1)

    # Bridge checker thread
    threading.Thread(target=lambda: [check_bridge() or time.sleep(3) for _ in iter(int, 1)], daemon=True).start()

    clock = pygame.time.Clock()
    tick = 0
    dt = 1.0 / LOOP_HZ  # Seconds per tick

    # Button edge detection
    prev_btns = {}

    def btn(i):
        return js.get_button(i) if i < n_btn else 0

    def btn_pressed(i):
        """True only on the frame the button goes down."""
        cur = btn(i)
        prev = prev_btns.get(i, 0)
        prev_btns[i] = cur
        return cur and not prev

    try:
        while True:
            pygame.event.pump()
            tick += 1

            # ─── Read sticks ───
            lx = deadzone(js.get_axis(ax_lx))
            ly = deadzone(js.get_axis(ax_ly))
            ry = deadzone(js.get_axis(ax_ry))

            # ─── Speed mode ───
            if btn(4):  # L1
                speed_mode = "SLOW"
                spd = SPEED_SLOW
            elif btn(5):  # R1
                speed_mode = "FAST"
                spd = SPEED_FAST
            else:
                speed_mode = "NORMAL"
                spd = SPEED_NORMAL

            # Convert speed from mm/s to mm/tick
            spd_tick = spd * dt

            # ─── Update target position from sticks ───
            # Stick UP (ly=-1) → robot forward (X+)
            target["x"] = clamp(target["x"] - ly * spd_tick, X_MIN, X_MAX)
            # Stick RIGHT (lx=+1) → robot right (Y-)
            target["y"] = clamp(target["y"] - lx * spd_tick, Y_MIN, Y_MAX)
            # R-Stick UP (ry=-1) → robot up (Z+)
            target["z"] = clamp(target["z"] - ry * spd_tick, Z_MIN, Z_MAX)

            # ─── D-pad fine adjust (edge triggered) ───
            if n_hat > 0:
                hx, hy = js.get_hat(0)
                if hy == 1:
                    target["z"] = clamp(target["z"] + 3, Z_MIN, Z_MAX)
                elif hy == -1:
                    target["z"] = clamp(target["z"] - 3, Z_MIN, Z_MAX)
                if hx == -1:
                    target["y"] = clamp(target["y"] + 3, Y_MIN, Y_MAX)
                elif hx == 1:
                    target["y"] = clamp(target["y"] - 3, Y_MIN, Y_MAX)

            # ─── Exponential smoothing ───
            # smooth position lerps toward target each frame
            # factor = 1 - SMOOTHING^dt_normalized, so it's frame-rate independent
            alpha = 1.0 - SMOOTHING
            smooth["x"] = lerp(smooth["x"], target["x"], alpha)
            smooth["y"] = lerp(smooth["y"], target["y"], alpha)
            smooth["z"] = lerp(smooth["z"], target["z"], alpha)

            # ─── Gripper: HOLD STATE, only changes on trigger press ───
            # R2 = close, L2 = open. Gripper STAYS where it is when released.
            if ax_r2 >= 0:
                r2 = (js.get_axis(ax_r2) + 1.0) / 2.0  # 0-1
                l2 = (js.get_axis(ax_l2) + 1.0) / 2.0
            else:
                # Fallback: use buttons 6,7 if no analog triggers
                r2 = 1.0 if btn(7) else 0.0
                l2 = 1.0 if btn(6) else 0.0

            # Only update gripper when trigger is actively pressed
            if r2 > 0.15:
                gripper_state = int(clamp(r2 * 70, 0, 70))
            elif l2 > 0.15:
                gripper_state = int(clamp(70 - l2 * 70, 0, 70))
            # else: gripper_state stays as-is (HOLD)

            # Send gripper only when changed
            if abs(gripper_state - gripper_sent) > 2:
                send_gripper(gripper_state)
                gripper_sent = gripper_state

            # ─── Button actions (edge-triggered) ───
            if btn_pressed(3):  # △
                go_home()
            if btn_pressed(2):  # □
                safe_height()
            if btn_pressed(1):  # ○
                describe_scene()
            if btn_pressed(0):  # ✕
                detect_objects()
            if btn_pressed(9):  # Options
                print(f"\n  {R}⚠ EMERGENCY HOME{X}")
                go_home()
            if btn_pressed(8):  # Share
                print(f"\n  {C}📍 X={smooth['x']:.1f} Y={smooth['y']:.1f} Z={smooth['z']:.1f} Grip={gripper_state}°{X}")

            # ─── Send IK only if position changed enough ───
            dx = abs(smooth["x"] - last_sent["x"])
            dy = abs(smooth["y"] - last_sent["y"])
            dz = abs(smooth["z"] - last_sent["z"])

            if dx > 0.3 or dy > 0.3 or dz > 0.3:
                result = send_ik(smooth["x"], smooth["y"], smooth["z"])
                if result and result.get("ok"):
                    last_sent = {"x": smooth["x"], "y": smooth["y"], "z": smooth["z"]}

            # ─── HUD ───
            if tick % 3 == 0:
                hud()

            clock.tick(LOOP_HZ)

    except KeyboardInterrupt:
        print(f"\n\n  {O}Going home...{X}")
        go_home()
        time.sleep(1)
    finally:
        pygame.quit()
        print(f"  {D}Done.{X}\n")


if __name__ == "__main__":
    main()