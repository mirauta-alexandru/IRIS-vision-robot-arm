#!/usr/bin/env python3
"""
IRIS 6-DOF Robotic Arm — PS4 Controller
=========================================

Mapping ANALOG (stick-uri — mișcare continuă cu ramp-up):
  Stick stâng X  → CH6 (Base)         — rotație stânga/dreapta
  Stick stâng Y  → CH4 (Shoulder A)   — sus/jos brațul
  Stick drept X  → CH2 (Wrist Pitch)  — sus/jos mâna
  Stick drept Y  → CH3 (Elbow)        — îndoaie/întinde cotul

Mapping INCREMENTAL (triggere/bumpers — adaugă/scad grade, ȚIN POZIȚIA):
  R2 = CH1 +grad   L2 = CH1 -grad   (Wrist Roll)
  R1 = CH0 +grad   L1 = CH0 -grad   (Gripper)

Butoane:
  ✕  = Home (lent)     ○  = Start sequence
  □  = Toggle gripper   △  = Show state
  OPTIONS = Speed+      SHARE = Speed-
  D-pad ↑↓ = CH2 Wrist pitch ±5°

Cerințe: pip install pygame pyserial
Utilizare: python3 iris_xbox_controller.py --port /dev/cu.usbserial-0001
"""

import time
import sys
import argparse
import math

try:
    import pygame
except ImportError:
    print("❌ pygame nu e instalat. Rulează: pip install pygame")
    sys.exit(1)

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ============================================================
# SERVO CONFIG (identic cu iris_analytic_ik.py)
# ============================================================
SERVO = {
    0: {"name": "Gripper",     "offset": 60,  "min": 0,   "max": 120},
    1: {"name": "Wrist Roll",  "offset": 55,  "min": 0,   "max": 180},
    2: {"name": "Wrist Pitch", "offset": 90,  "min": 0,   "max": 170},
    3: {"name": "Elbow",       "offset": 50,  "min": 20,  "max": 180},
    4: {"name": "Shoulder A",  "offset": 150, "min": 0,   "max": 150},
    5: {"name": "Shoulder B",  "offset": 30,  "min": 30,  "max": 120},
    6: {"name": "Base",        "offset": 90,  "min": 0,   "max": 180},
}

HOME = {0: 90, 1: 55, 2: 150, 3: 180, 4: 0, 5: 180, 6: 73}

START_STEPS = [
    {0: 60, 1: 55, 2: 130, 3: 60, 4: 80, 5: 100, 6: 90},
    {0: 60, 1: 55, 2: 110, 3: 80, 4: 75, 5: 105, 6: 90},
]

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ============================================================
# SERIAL CONTROLLER
# ============================================================

class SerialController:
    def __init__(self, port, baud=115200):
        if not HAS_SERIAL:
            raise RuntimeError("pyserial not installed: pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)
        self.ser.reset_input_buffer()
        print(f"✅ Conectat la {port}")

    def send(self, channel, angle):
        cmd = f"ch {channel} {int(angle)}\n"
        self.ser.write(cmd.encode())
        time.sleep(0.01)

    def send_all_smooth(self, target, current, steps=40, duration=2.0):
        """Mișcare smooth de la current la target (interpolare liniară)."""
        channels = sorted(k for k in target.keys() if isinstance(k, int) and k != 5)
        step_delay = duration / steps
        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)  # ease in-out
            for ch in channels:
                val = current.get(ch, target[ch]) + (target[ch] - current.get(ch, target[ch])) * t
                cmd = f"ch {ch} {int(val)}\n"
                self.ser.write(cmd.encode())
            time.sleep(step_delay)

    def close(self):
        self.ser.close()
        print("🔌 Serial închis")


class DummyController:
    def send(self, channel, angle): pass
    def send_all_smooth(self, target, current, steps=40, duration=2.0): pass
    def close(self): pass


# ============================================================
# DEADZONE & RAMP-UP
# ============================================================

DEADZONE = 0.15

def apply_deadzone(value):
    if abs(value) < DEADZONE:
        return 0.0
    sign = 1 if value > 0 else -1
    return sign * (abs(value) - DEADZONE) / (1.0 - DEADZONE)


class RampUp:
    """Ramp-up exponențial: 15% → 100% în ~1.8s. Reset la eliberare."""
    def __init__(self, min_factor=0.15, tau=0.6):
        self.min_factor = min_factor
        self.tau = tau
        self.active_since = {}

    def get_factor(self, key, is_active):
        now = time.time()
        if not is_active:
            self.active_since.pop(key, None)
            return 0.0
        if key not in self.active_since:
            self.active_since[key] = now
        elapsed = now - self.active_since[key]
        return self.min_factor + (1.0 - self.min_factor) * (1.0 - math.exp(-elapsed / self.tau))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="IRIS PS4 Controller")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--speed", type=float, default=0.4,
                        help="Max grade/frame pentru stick-uri (default 0.4)")
    parser.add_argument("--trigger-speed", type=float, default=0.3,
                        help="Grade/frame pentru L2/R2/L1/R1 (default 0.3)")
    args = parser.parse_args()

    if args.port:
        ctrl = SerialController(args.port, args.baud)
    else:
        print("ℹ️  Mod simulare (fără serial)")
        ctrl = DummyController()

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("❌ Niciun controller detectat!")
        ctrl.close()
        sys.exit(1)

    joy = pygame.joystick.Joystick(0)
    joy.init()
    name = joy.get_name()
    num_btns = joy.get_numbuttons()
    num_axes = joy.get_numaxes()
    num_hats = joy.get_numhats()
    print(f"🎮 Controller: {name}")
    print(f"   Axe: {num_axes}, Butoane: {num_btns}, Haturi: {num_hats}")

    # Detect PS4 vs Xbox
    is_ps4 = num_btns > 10
    print(f"   Detectat: {'PS4' if is_ps4 else 'Xbox'}")

    # PS4 SDL2 button mapping:
    #  0=✕  1=○  2=□  3=△
    #  4=L1  5=R1  6=L2btn  7=R2btn
    #  8=Share  9=Options  10=PS  11=L3  12=R3
    #  13=D-Up  14=D-Down  15=D-Left  16=D-Right
    #
    # PS4 SDL2 axis mapping:
    #  0=LeftStickX  1=LeftStickY  2=RightStickX  3=RightStickY
    #  4=L2(trigger)  5=R2(trigger)
    #  Triggers: -1=released, +1=full press
    #
    # Xbox SDL2:
    #  Buttons: 0=A 1=B 2=X 3=Y 4=LB 5=RB 6=Back 7=Start
    #  Axes: 0=LX 1=LY 2=LT 3=RX 4=RY 5=RT

    # Button indices
    if is_ps4:
        BTN_CROSS = 0; BTN_CIRCLE = 1; BTN_SQUARE = 2; BTN_TRIANGLE = 3
        BTN_L1 = 4; BTN_R1 = 5
        BTN_SHARE = 8; BTN_OPTIONS = 9
        BTN_DUP = 13; BTN_DDOWN = 14
        # Axes
        AX_LX = 0; AX_LY = 1; AX_RX = 2; AX_RY = 3
        AX_L2 = 4; AX_R2 = 5
    else:
        BTN_CROSS = 0; BTN_CIRCLE = 1; BTN_SQUARE = 2; BTN_TRIANGLE = 3
        BTN_L1 = 4; BTN_R1 = 5
        BTN_SHARE = 6; BTN_OPTIONS = 7
        BTN_DUP = -1; BTN_DDOWN = -1  # Xbox uses hats
        AX_LX = 0; AX_LY = 1; AX_RX = 3; AX_RY = 4
        AX_L2 = 2; AX_R2 = 5

    current_angles = dict(HOME)
    speed = args.speed
    trigger_speed = args.trigger_speed
    gripper_open = True
    fps = 60
    clock = pygame.time.Clock()
    ramp = RampUp(min_factor=0.15, tau=0.6)
    last_sent = {}
    last_print = 0.0

    screen = pygame.display.set_mode((1, 1))
    pygame.display.set_caption("IRIS")

    print()
    print("🤖 IRIS Controller — ACTIV")
    print("=" * 50)
    print(f"  Stick speed: {speed}°/f ({speed*fps:.0f}°/s max)")
    print(f"  Trigger speed: {trigger_speed}°/f ({trigger_speed*fps:.0f}°/s max)")
    print()
    for ch in sorted(SERVO.keys()):
        s = SERVO[ch]
        print(f"  CH{ch} {s['name']:14s} [{s['min']:3d}° - {s['max']:3d}°]  home={HOME[ch]}°")
    print()
    print("  Stick L X/Y  → CH6 Base / CH4 Shoulder")
    print("  Stick R X    → CH2 Wrist Pitch")
    print("  Stick R Y    → CH3 Elbow")
    print("  L2(-)/R2(+)  → CH1 Wrist Roll   (incremental, ține poziția)")
    print("  L1(-)/R1(+)  → CH0 Gripper       (incremental, ține poziția)")
    print("  ✕=Home  ○=Start  □=ToggleGrip  △=Show")
    print("  OPTIONS=Speed+  SHARE=Speed-  D-pad↑↓=CH2±5°")
    print()

    # Trimite home LENT
    ctrl.send_all_smooth(current_angles, current_angles, steps=1, duration=0.1)
    print(f"🏠 Home trimis")

    running = True
    try:
        while running:
            clock.tick(fps)
            changed = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.JOYBUTTONDOWN:
                    btn = event.button

                    if btn == BTN_CROSS:  # ✕ — Home LENT
                        print("\n🏠 Home (smooth)...")
                        old = dict(current_angles)
                        current_angles = dict(HOME)
                        ctrl.send_all_smooth(current_angles, old, steps=50, duration=3.0)
                        last_sent.clear()
                        print("🏠 Home done")

                    elif btn == BTN_CIRCLE:  # ○ — Start sequence
                        print("\n🚀 Start sequence")
                        old = dict(current_angles)
                        ctrl.send_all_smooth(START_STEPS[0], old, steps=40, duration=2.0)
                        ctrl.send_all_smooth(START_STEPS[1], START_STEPS[0], steps=40, duration=2.0)
                        current_angles = dict(START_STEPS[-1])
                        last_sent.clear()
                        print("✅ Ready")

                    elif btn == BTN_SQUARE:  # □ — Toggle gripper
                        gripper_open = not gripper_open
                        current_angles[0] = clamp(
                            30 if gripper_open else 120,
                            SERVO[0]["min"], SERVO[0]["max"]
                        )
                        ctrl.send(0, current_angles[0])
                        print(f"\n{'🤚 OPEN' if gripper_open else '✊ CLOSED'} ch0={current_angles[0]:.0f}°")

                    elif btn == BTN_TRIANGLE:  # △ — Show
                        print(f"\n📐 " + " ".join(
                            f"ch{i}={current_angles[i]:.1f}°" for i in range(7)
                        ))
                        print(f"   Stick speed: {speed:.2f}  Trigger speed: {trigger_speed:.2f}")

                    elif btn == BTN_OPTIONS:
                        speed = min(speed + 0.05, 2.0)
                        trigger_speed = min(trigger_speed + 0.05, 1.5)
                        print(f"\n🏎️  Stick={speed:.2f} Trigger={trigger_speed:.2f}")

                    elif btn == BTN_SHARE:
                        speed = max(speed - 0.05, 0.05)
                        trigger_speed = max(trigger_speed - 0.05, 0.05)
                        print(f"\n🐌 Stick={speed:.2f} Trigger={trigger_speed:.2f}")

                    # PS4 D-pad as buttons
                    elif btn == BTN_DUP:
                        current_angles[2] = clamp(current_angles[2] + 5, SERVO[2]["min"], SERVO[2]["max"])
                        ctrl.send(2, current_angles[2])
                        print(f"\n⬆️ CH2 → {current_angles[2]:.0f}°")
                    elif btn == BTN_DDOWN:
                        current_angles[2] = clamp(current_angles[2] - 5, SERVO[2]["min"], SERVO[2]["max"])
                        ctrl.send(2, current_angles[2])
                        print(f"\n⬇️ CH2 → {current_angles[2]:.0f}°")

                # Xbox D-pad as hat
                elif event.type == pygame.JOYHATMOTION:
                    _, hat_y = event.value
                    if hat_y == 1:
                        current_angles[2] = clamp(current_angles[2] + 5, SERVO[2]["min"], SERVO[2]["max"])
                        ctrl.send(2, current_angles[2])
                        print(f"\n⬆️ CH2 → {current_angles[2]:.0f}°")
                    elif hat_y == -1:
                        current_angles[2] = clamp(current_angles[2] - 5, SERVO[2]["min"], SERVO[2]["max"])
                        ctrl.send(2, current_angles[2])
                        print(f"\n⬇️ CH2 → {current_angles[2]:.0f}°")

            # ================================================
            # STICK-URI ANALOGICE (mișcare continuă cu ramp-up)
            # ================================================

            # CH6 — Base [0°-180°] — Stick stâng X
            lx = apply_deadzone(joy.get_axis(AX_LX))
            rf = ramp.get_factor("ch6", lx != 0)
            if lx != 0:
                current_angles[6] = clamp(
                    current_angles[6] + lx * speed * rf,
                    SERVO[6]["min"], SERVO[6]["max"]
                )
                changed = True

            # CH4 — Shoulder [0°-150°] — Stick stâng Y (sus=crește)
            ly = apply_deadzone(joy.get_axis(AX_LY))
            rf = ramp.get_factor("ch4", ly != 0)
            if ly != 0:
                current_angles[4] = clamp(
                    current_angles[4] - ly * speed * rf,
                    SERVO[4]["min"], SERVO[4]["max"]
                )
                current_angles[5] = clamp(180 - current_angles[4], SERVO[5]["min"], SERVO[5]["max"])
                changed = True

            # CH2 — Wrist Pitch [0°-170°] — Stick drept X
            rx = apply_deadzone(joy.get_axis(AX_RX))
            rf = ramp.get_factor("ch2", rx != 0)
            if rx != 0:
                current_angles[2] = clamp(
                    current_angles[2] + rx * speed * rf,
                    SERVO[2]["min"], SERVO[2]["max"]
                )
                changed = True

            # CH3 — Elbow [20°-180°] — Stick drept Y (sus=crește)
            ry = apply_deadzone(joy.get_axis(AX_RY))
            rf = ramp.get_factor("ch3", ry != 0)
            if ry != 0:
                current_angles[3] = clamp(
                    current_angles[3] - ry * speed * rf,
                    SERVO[3]["min"], SERVO[3]["max"]
                )
                changed = True

            # ================================================
            # TRIGGERE ANALOGICE — INCREMENTAL (ȚIN POZIȚIA!)
            # L2/R2 → CH1, L1/R1 → CH0
            # Trigger value: -1=released, +1=full press
            # Convertim la 0..1, apoi aplicăm ca increment
            # ================================================

            if num_axes > AX_L2 and num_axes > AX_R2:
                # --- CH1 Wrist Roll: L2 = scade, R2 = crește ---
                l2_raw = joy.get_axis(AX_L2)
                r2_raw = joy.get_axis(AX_R2)
                l2 = max(0, (l2_raw + 1) / 2.0)  # 0..1
                r2 = max(0, (r2_raw + 1) / 2.0)   # 0..1

                if r2 > 0.05:
                    rf = ramp.get_factor("ch1+", True)
                    current_angles[1] = clamp(
                        current_angles[1] + r2 * trigger_speed * rf,
                        SERVO[1]["min"], SERVO[1]["max"]
                    )
                    changed = True
                else:
                    ramp.get_factor("ch1+", False)

                if l2 > 0.05:
                    rf = ramp.get_factor("ch1-", True)
                    current_angles[1] = clamp(
                        current_angles[1] - l2 * trigger_speed * rf,
                        SERVO[1]["min"], SERVO[1]["max"]
                    )
                    changed = True
                else:
                    ramp.get_factor("ch1-", False)

            # --- CH0 Gripper: L1 = scade, R1 = crește ---
            # L1/R1 sunt butoane digitale, dar ții apăsat = repeta la fiecare frame
            l1_pressed = joy.get_button(BTN_L1) if num_btns > BTN_L1 else False
            r1_pressed = joy.get_button(BTN_R1) if num_btns > BTN_R1 else False

            if r1_pressed:
                rf = ramp.get_factor("ch0+", True)
                current_angles[0] = clamp(
                    current_angles[0] + trigger_speed * rf,
                    SERVO[0]["min"], SERVO[0]["max"]
                )
                changed = True
            else:
                ramp.get_factor("ch0+", False)

            if l1_pressed:
                rf = ramp.get_factor("ch0-", True)
                current_angles[0] = clamp(
                    current_angles[0] - trigger_speed * rf,
                    SERVO[0]["min"], SERVO[0]["max"]
                )
                changed = True
            else:
                ramp.get_factor("ch0-", False)

            # --- Trimite doar ce s-a schimbat cu ≥1° ---
            if changed:
                for ch in [0, 1, 2, 3, 4, 6]:
                    val = int(current_angles[ch])
                    if val != last_sent.get(ch, -999):
                        ctrl.send(ch, val)
                        last_sent[ch] = val

            # --- Terminal status ---
            now = time.time()
            if changed and now - last_print > 0.3:
                status = " ".join(
                    f"ch{i}={current_angles[i]:5.1f}" for i in [6, 4, 3, 2, 1, 0]
                )
                print(f"\r🎮 {status}", end="", flush=True)
                last_print = now

    except KeyboardInterrupt:
        print("\n⏹️  Ctrl+C")
    finally:
        ctrl.close()
        pygame.quit()
        print("👋 Bye!")


if __name__ == "__main__":
    main()
