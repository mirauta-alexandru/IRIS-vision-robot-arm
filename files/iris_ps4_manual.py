#!/usr/bin/env python3
"""
IRIS PS4 Manual Control

Fresh controller script for manual servo driving through iris_bridge.py.
It does not talk to the ESP32 directly; all motion goes through the bridge HTTP
API so the same smoothing, limits, and base compensation stay active.
"""

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass

import requests


DEFAULT_BRIDGE = "http://localhost:8765"
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
pygame = None

GRIPPER_OPEN_DEG = 0.0
GRIPPER_CLOSED_DEG = 70.0

DEFAULT_SERVO_STATE = {
    0: 0.0,     # gripper
    1: 55.0,    # wrist roll
    2: 100.0,   # wrist pitch
    3: 25.0,    # elbow
    4: 160.0,   # shoulder A
    6: 75.0,    # base rotation
}

SERVO_LIMITS = {
    0: (0.0, 70.0),
    1: (0.0, 180.0),
    2: (0.0, 170.0),
    3: (20.0, 180.0),
    4: (30.0, 160.0),
    6: (0.0, 180.0),
}


@dataclass
class AxisControl:
    ch: int
    axis: int
    name: str
    invert: bool = False


AXIS_CONTROLS = [
    AxisControl(ch=3, axis=3, name="right stick Y -> CH3", invert=True),
    AxisControl(ch=2, axis=2, name="right stick X -> CH2"),
    AxisControl(ch=6, axis=0, name="left stick X -> CH6", invert=True),
    AxisControl(ch=4, axis=1, name="left stick Y -> CH4", invert=True),
]

# Pygame button ids vary a bit between macOS / SDL mappings, so we accept the
# two common PS4 layouts. Override from CLI if your controller reports another.
L1_BUTTON_CANDIDATES = (4, 9)
R1_BUTTON_CANDIDATES = (5, 10)

L2_AXIS = 4
R2_AXIS = 5


running = True


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * ((abs(value) - deadzone) / (1.0 - deadzone))


def normalize_trigger(value):
    # Some controllers idle at -1 and press to +1; others idle at 0 and press to +1.
    if value < 0:
        return clamp((value + 1.0) * 0.5, 0.0, 1.0)
    return clamp(value, 0.0, 1.0)


def joystick_axis(js, index):
    if index < 0 or index >= js.get_numaxes():
        return 0.0
    return float(js.get_axis(index))


def any_button(js, indices):
    for index in indices:
        if 0 <= index < js.get_numbuttons() and js.get_button(index):
            return True
    return False


class BridgeClient:
    def __init__(self, bridge_url, timeout=0.4):
        self.bridge_url = bridge_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.last_sent = {}

    def load_current_state(self):
        state = dict(DEFAULT_SERVO_STATE)
        try:
            res = self.session.get(f"{self.bridge_url}/pose", timeout=self.timeout)
            data = res.json()
            for ch_str, angle in data.get("servos", {}).items():
                ch = int(ch_str)
                if ch in state:
                    state[ch] = float(angle)
        except Exception as exc:
            print(f"Bridge state unavailable, using defaults: {exc}")
        return state

    def send_servo(self, ch, angle):
        lo, hi = SERVO_LIMITS[ch]
        angle = round(clamp(angle, lo, hi), 1)
        previous = self.last_sent.get(ch)
        if previous is not None and abs(previous - angle) < 0.5:
            return

        try:
            if ch == 0:
                self.session.get(
                    f"{self.bridge_url}/gripper",
                    params={"angle": int(round(angle))},
                    timeout=self.timeout,
                )
            elif ch == 4:
                # Shoulder A needs CH5 mirrored, same as the visualizer does.
                mirror = int(round(180.0 - angle))
                self.session.post(
                    f"{self.bridge_url}/servos",
                    json={"4": int(round(angle)), "5": mirror},
                    timeout=self.timeout,
                )
            else:
                self.session.get(
                    f"{self.bridge_url}/servo",
                    params={"ch": ch, "angle": int(round(angle))},
                    timeout=self.timeout,
                )
            self.last_sent[ch] = angle
        except Exception as exc:
            print(f"Bridge send failed for CH{ch}: {exc}")


def handle_signal(_signum, _frame):
    global running
    running = False


def wait_for_controller():
    print("Waiting for PS4 controller...")
    while running:
        pygame.event.pump()
        pygame.joystick.quit()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            js = pygame.joystick.Joystick(0)
            js.init()
            return js
        time.sleep(0.5)
    return None


def print_mapping(js, args):
    print()
    print("IRIS PS4 Manual Control")
    print(f"Controller: {js.get_name()}")
    print(f"Axes: {js.get_numaxes()} | Buttons: {js.get_numbuttons()}")
    print()
    print("Mapping:")
    print("  R1              -> open gripper (CH0 = 0)")
    print("  L1              -> close gripper (CH0 = 70)")
    print("  R2 / L2         -> wrist roll CH1 gradually")
    print("  Right stick Y   -> CH3")
    print("  Right stick X   -> CH2")
    print("  Left stick X    -> CH6 inverted")
    print("  Left stick Y    -> CH4 + CH5 mirror")
    print()
    print(f"Bridge: {args.bridge}")
    print("Press Ctrl+C to stop.")
    print()


def main():
    global pygame
    parser = argparse.ArgumentParser(description="Manual PS4 controller for IRIS through iris_bridge.py")
    parser.add_argument("--bridge", default=DEFAULT_BRIDGE, help="iris_bridge URL")
    parser.add_argument("--rate-hz", type=float, default=35.0, help="control loop rate")
    parser.add_argument("--servo-speed", type=float, default=70.0, help="servo degrees per second at full stick")
    parser.add_argument("--wrist-roll-speed", type=float, default=110.0, help="CH1 degrees per second at full trigger")
    parser.add_argument("--deadzone", type=float, default=0.12, help="stick deadzone")
    parser.add_argument("--l1-button", type=int, default=None, help="override L1 pygame button id")
    parser.add_argument("--r1-button", type=int, default=None, help="override R1 pygame button id")
    args = parser.parse_args()

    try:
        import pygame as pygame_module
    except ModuleNotFoundError:
        print("pygame is not installed. Run: pip install -r requirements.txt")
        return 1
    pygame = pygame_module

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    pygame.init()
    pygame.joystick.init()

    js = wait_for_controller()
    if js is None:
        return 0

    print_mapping(js, args)

    bridge = BridgeClient(args.bridge)
    state = bridge.load_current_state()
    for ch, angle in state.items():
        bridge.last_sent[ch] = round(angle, 1)

    l1_buttons = (args.l1_button,) if args.l1_button is not None else L1_BUTTON_CANDIDATES
    r1_buttons = (args.r1_button,) if args.r1_button is not None else R1_BUTTON_CANDIDATES

    last_l1 = False
    last_r1 = False
    last_report = 0.0
    clock = pygame.time.Clock()

    while running:
        dt = clock.tick(args.rate_hz) / 1000.0
        pygame.event.pump()

        l1 = any_button(js, l1_buttons)
        r1 = any_button(js, r1_buttons)

        if r1 and not last_r1:
            state[0] = GRIPPER_OPEN_DEG
            bridge.send_servo(0, state[0])
            print("R1 -> gripper open")
        if l1 and not last_l1:
            state[0] = GRIPPER_CLOSED_DEG
            bridge.send_servo(0, state[0])
            print("L1 -> gripper closed")
        last_l1 = l1
        last_r1 = r1

        r2 = normalize_trigger(joystick_axis(js, R2_AXIS))
        l2 = normalize_trigger(joystick_axis(js, L2_AXIS))
        wrist_roll_delta = (r2 - l2) * args.wrist_roll_speed * dt
        if abs(wrist_roll_delta) > 0.05:
            state[1] = clamp(state[1] + wrist_roll_delta, *SERVO_LIMITS[1])
            bridge.send_servo(1, state[1])

        for control in AXIS_CONTROLS:
            raw = joystick_axis(js, control.axis)
            value = apply_deadzone(raw, args.deadzone)
            if control.invert:
                value *= -1.0
            if value == 0.0:
                continue
            state[control.ch] = clamp(
                state[control.ch] + value * args.servo_speed * dt,
                *SERVO_LIMITS[control.ch],
            )
            bridge.send_servo(control.ch, state[control.ch])

        now = time.time()
        if now - last_report > 1.0:
            last_report = now
            print(f"CH0={state[0]:.0f} CH1={state[1]:.0f} CH2={state[2]:.0f} CH3={state[3]:.0f} CH4={state[4]:.0f} CH6={state[6]:.0f}")

    pygame.quit()
    print("PS4 manual control stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
