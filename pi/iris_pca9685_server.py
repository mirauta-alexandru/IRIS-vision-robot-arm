#!/usr/bin/env python3
"""IRIS Raspberry Pi PCA9685 controller.

The Dell runs vision, AI, and IK. This process runs on the Raspberry Pi and
accepts authenticated joint targets over the trusted local network.

Hardware movement is OFF unless BOTH conditions are met:
  1. IRIS_ENABLE_HARDWARE=1
  2. For normal angle commands, all seven channels are enabled and calibrated.
     Single-channel pulse jogging additionally requires IRIS_CALIBRATION_MODE=1.

This is a prototype controller, not a safety-rated emergency stop.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import socketserver
from pathlib import Path
from typing import Any

import board
import busio
from adafruit_pca9685 import PCA9685

HOST = os.environ.get("IRIS_BIND_HOST", "0.0.0.0")
PORT = int(os.environ.get("IRIS_PI_PORT", "8766"))
ALLOWED_PC_IP = os.environ.get("IRIS_ALLOWED_PC_IP", "192.168.1.213")
TOKEN = os.environ.get("IRIS_TOKEN", "")
HARDWARE_ENABLED = os.environ.get("IRIS_ENABLE_HARDWARE", "0") == "1"
CALIBRATION_MODE = os.environ.get("IRIS_CALIBRATION_MODE", "0") == "1"
COORDINATED_MOTION_ENABLED = os.environ.get("IRIS_ALLOW_COORDINATED_MOTION", "0") == "1"
CONFIG_PATH = Path(__file__).with_name("servo_calibration.json")
MAX_LINE_BYTES = 4096

if not TOKEN:
    raise RuntimeError("IRIS_TOKEN is required. Set the same secret on the Dell and Pi.")

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)

FREQUENCY_HZ = int(CONFIG.get("frequency_hz", 50))
CHANNELS = CONFIG.get("channels", {})
if not isinstance(CHANNELS, dict):
    raise RuntimeError("servo_calibration.json must contain a channels object.")

pca: PCA9685 | None = None
STOP_LATCHED = False
if HARDWARE_ENABLED:
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = FREQUENCY_HZ
    print(f"IRIS PCA9685 ready at {FREQUENCY_HZ} Hz; hardware gate is ON")
else:
    print("IRIS PCA9685 server started in DRY-RUN mode; no PWM will be sent")


def response(status: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "hardware_enabled": HARDWARE_ENABLED,
        "hardware_moved": False,
        **extra,
    }


def get_channel_config(channel: int) -> dict[str, Any]:
    if not 0 <= channel <= 15:
        raise ValueError("PCA9685 channel must be between 0 and 15.")
    item = CHANNELS.get(str(channel))
    if not isinstance(item, dict):
        raise ValueError(f"Channel {channel} has no calibration entry.")
    if item.get("enabled") is not True:
        raise PermissionError(f"Channel {channel} is disabled pending calibration.")
    if item.get("calibration_verified") is not True:
        raise PermissionError(f"Channel {channel} calibration has not been explicitly verified.")
    pulse_min = item.get("pulse_min_us")
    pulse_max = item.get("pulse_max_us")
    if not isinstance(pulse_min, (int, float)) or not isinstance(pulse_max, (int, float)):
        raise ValueError(f"Channel {channel} has no calibrated pulse limits.")
    if not 400 <= pulse_min < pulse_max <= 2600:
        raise ValueError(f"Channel {channel} pulse limits are invalid.")
    if item.get("pulse_direction", "increasing") not in ("increasing", "decreasing"):
        raise ValueError(f"Channel {channel} pulse direction must be increasing or decreasing.")
    return item


def pulse_to_duty_cycle(pulse_us: float) -> int:
    period_us = 1_000_000.0 / FREQUENCY_HZ
    if not 0 < pulse_us < period_us:
        raise ValueError("Pulse width is outside the configured PWM period.")
    return round((pulse_us / period_us) * 65535)


def set_pulse(channel: int, pulse_us: float) -> dict[str, Any]:
    if STOP_LATCHED:
        raise PermissionError("STOP is latched. Restart the Pi controller deliberately before issuing more commands.")
    if not 0 <= channel <= 15:
        raise ValueError("PCA9685 channel must be between 0 and 15.")
    if not math.isfinite(pulse_us):
        raise ValueError("Pulse width must be a finite number.")
    # Dry-run requests must remain testable while every physical channel is
    # disabled and uncalibrated. Never initialize I2C or write PWM in this path.
    if not HARDWARE_ENABLED:
        return response("dry_run", channel=channel, pulse_us=pulse_us)
    item = get_channel_config(channel)
    pulse_min = float(item["pulse_min_us"])
    pulse_max = float(item["pulse_max_us"])
    if not pulse_min <= pulse_us <= pulse_max:
        raise ValueError(
            f"Pulse {pulse_us:g} us is outside channel {channel} limits "
            f"({pulse_min:g}-{pulse_max:g} us)."
        )
    if pca is None:
        raise RuntimeError("PCA9685 hardware is not initialized.")
    pca.channels[channel].duty_cycle = pulse_to_duty_cycle(pulse_us)
    return response(
        "moved",
        hardware_moved=True,
        channel=channel,
        pulse_us=round(pulse_us, 1),
    )


def set_angle(channel: int, angle: float) -> dict[str, Any]:
    if STOP_LATCHED:
        raise PermissionError("STOP is latched. Restart the Pi controller deliberately before issuing more commands.")
    if not 0 <= channel <= 15:
        raise ValueError("PCA9685 channel must be between 0 and 15.")
    if not math.isfinite(angle):
        raise ValueError("Angle must be a finite number.")
    # The Dell bridge can exercise its complete command path in dry-run mode
    # before any servo calibration exists. This branch cannot move hardware.
    if not HARDWARE_ENABLED:
        return response("dry_run", channel=channel, angle=round(angle, 2))
    item = get_channel_config(channel)
    angle_min = float(item["angle_min"])
    angle_max = float(item["angle_max"])
    if angle_min >= angle_max:
        raise ValueError(f"Channel {channel} angle limits are invalid.")
    if not angle_min <= angle <= angle_max:
        raise ValueError(
            f"Angle {angle:g} degrees is outside channel {channel} limits "
            f"({angle_min:g}-{angle_max:g})."
        )
    pulse_min = float(item["pulse_min_us"])
    pulse_max = float(item["pulse_max_us"])
    ratio = (angle - angle_min) / (angle_max - angle_min)
    if item.get("pulse_direction", "increasing") == "decreasing":
        pulse_us = pulse_max - ratio * (pulse_max - pulse_min)
    else:
        pulse_us = pulse_min + ratio * (pulse_max - pulse_min)
    result = set_pulse(channel, pulse_us)
    result["angle"] = round(angle, 2)
    return result


def set_angles(angles: dict[str, Any]) -> dict[str, Any]:
    """Validate a coordinated multi-joint target completely before writing PWM."""
    if STOP_LATCHED:
        raise PermissionError("STOP is latched. Restart the Pi controller deliberately before issuing more commands.")
    if not isinstance(angles, dict) or not angles:
        raise ValueError("SET_ANGLES requires a non-empty channel-to-angle object.")
    normalized: dict[int, float] = {}
    for raw_channel, raw_angle in angles.items():
        channel = int(raw_channel)
        angle = float(raw_angle)
        if not 0 <= channel <= 6:
            raise ValueError("Coordinated arm targets are limited to channels 0-6.")
        if not math.isfinite(angle):
            raise ValueError(f"Channel {channel} angle must be finite.")
        normalized[channel] = angle
    if not HARDWARE_ENABLED:
        return response("dry_run", channels=normalized, hardware_moved=False)
    if not COORDINATED_MOTION_ENABLED:
        raise PermissionError("Coordinated motion is locked until calibration review.")
    if not all_arm_channels_calibrated():
        raise PermissionError("Coordinated motion is locked until all seven servo channels are calibrated.")
    # Build and validate the entire PWM plan before changing any channel.
    pwm_plan: dict[int, int] = {}
    for channel, angle in normalized.items():
        item = get_channel_config(channel)
        angle_min = float(item["angle_min"])
        angle_max = float(item["angle_max"])
        if not angle_min <= angle <= angle_max:
            raise ValueError(f"Channel {channel} angle {angle:g} is outside calibrated limits ({angle_min:g}-{angle_max:g}).")
        pulse_min = float(item["pulse_min_us"])
        pulse_max = float(item["pulse_max_us"])
        ratio = (angle - angle_min) / (angle_max - angle_min)
        if item.get("pulse_direction", "increasing") == "decreasing":
            pulse_us = pulse_max - ratio * (pulse_max - pulse_min)
        else:
            pulse_us = pulse_min + ratio * (pulse_max - pulse_min)
        pwm_plan[channel] = pulse_to_duty_cycle(pulse_us)
    if pca is None:
        raise RuntimeError("PCA9685 hardware is not initialized.")
    for channel, duty_cycle in pwm_plan.items():
        pca.channels[channel].duty_cycle = duty_cycle
    return response("moved", channels=normalized, hardware_moved=True)

def all_arm_channels_calibrated() -> bool:
    """Require all seven arm joints before accepting coordinated angle motion."""
    for channel in range(7):
        item = CHANNELS.get(str(channel))
        if not isinstance(item, dict) or item.get("enabled") is not True:
            return False
        if item.get("calibration_verified") is not True:
            return False
        pulse_min = item.get("pulse_min_us")
        pulse_max = item.get("pulse_max_us")
        angle_min = item.get("angle_min")
        angle_max = item.get("angle_max")
        if not isinstance(pulse_min, (int, float)) or not isinstance(pulse_max, (int, float)):
            return False
        if not 400 <= pulse_min < pulse_max <= 2600:
            return False
        if not isinstance(angle_min, (int, float)) or not isinstance(angle_max, (int, float)):
            return False
        if angle_min >= angle_max:
            return False
        if item.get("pulse_direction", "increasing") not in ("increasing", "decreasing"):
            return False
    return True


def calibration_jog(channel: int, pulse_us: float) -> dict[str, Any]:
    """Move one unloaded servo by a tightly bounded pulse during calibration only."""
    if STOP_LATCHED:
        raise PermissionError("STOP is latched. Restart the Pi controller deliberately before issuing more commands.")
    if not HARDWARE_ENABLED or not CALIBRATION_MODE or pca is None:
        raise PermissionError(
            "Calibration jog requires IRIS_ENABLE_HARDWARE=1 and IRIS_CALIBRATION_MODE=1."
        )
    if not 0 <= channel <= 6:
        raise ValueError("Calibration jog is limited to arm channels 0-6.")
    if not math.isfinite(pulse_us):
        raise ValueError("Pulse width must be a finite number.")
    item = CHANNELS.get(str(channel))
    if not isinstance(item, dict):
        raise ValueError(f"Channel {channel} has no calibration entry.")
    jog_min = item.get("jog_min_us")
    jog_max = item.get("jog_max_us")
    if not isinstance(jog_min, (int, float)) or not isinstance(jog_max, (int, float)):
        raise ValueError(f"Channel {channel} has no temporary jog window.")
    if not 400 <= jog_min < jog_max <= 2600:
        raise ValueError(f"Channel {channel} temporary jog window is invalid.")
    if not jog_min <= pulse_us <= jog_max:
        raise ValueError(
            f"Pulse {pulse_us:g} us is outside the temporary channel {channel} "
            f"jog window ({jog_min:g}-{jog_max:g} us)."
        )
    pca.channels[channel].duty_cycle = pulse_to_duty_cycle(pulse_us)
    return response(
        "calibration_jogged",
        hardware_moved=True,
        channel=channel,
        pulse_us=round(pulse_us, 1),
        temporary_jog=True,
    )


def handle_command(command: dict[str, Any]) -> dict[str, Any]:
    global STOP_LATCHED
    action = str(command.get("command", "")).upper()

    if action == "PING":
        return response(
            "ok",
            controller="raspberry_pi_pca9685",
            frequency_hz=FREQUENCY_HZ,
            calibrated_channels=[
                int(ch) for ch, cfg in CHANNELS.items()
                if isinstance(cfg, dict) and cfg.get("enabled") is True
            ],
            full_arm_ready=all_arm_channels_calibrated(),
            calibration_mode=CALIBRATION_MODE,
            coordinated_motion_enabled=COORDINATED_MOTION_ENABLED,
            stop_latched=STOP_LATCHED,
        )

    if action == "STOP":
        # Latch the command path but preserve the last PWM setpoints so a
        # gravity-loaded arm does not suddenly lose servo holding torque.
        STOP_LATCHED = True
        return response(
            "stopped",
            outputs_held=bool(HARDWARE_ENABLED),
            outputs_disabled=False,
            stop_latched=True,
            message="Last PWM setpoints are held. Commands are blocked until the Pi controller process is restarted.",
        )

    if action == "SET_ANGLE":
        if HARDWARE_ENABLED and not COORDINATED_MOTION_ENABLED:
            raise PermissionError(
                "Coordinated motion is locked. Set IRIS_ALLOW_COORDINATED_MOTION=1 only after calibration and review."
            )
        if HARDWARE_ENABLED and not all_arm_channels_calibrated():
            raise PermissionError(
                "Coordinated angle motion is locked until all seven servo channels are calibrated."
            )
        channel = int(command["channel"])
        angle = float(command["angle"])
        return set_angle(channel, angle)

    if action == "SET_ANGLES":
        return set_angles(command.get("angles", {}))
    if action == "SET_PULSE":
        if HARDWARE_ENABLED and not CALIBRATION_MODE:
            raise PermissionError(
                "Direct pulse commands require IRIS_CALIBRATION_MODE=1."
            )
        channel = int(command["channel"])
        pulse_us = float(command["pulse_us"])
        return set_pulse(channel, pulse_us)

    if action == "CALIBRATION_JOG":
        channel = int(command["channel"])
        pulse_us = float(command["pulse_us"])
        return calibration_jog(channel, pulse_us)

    raise ValueError(
        "Unsupported command. Use PING, STOP, SET_ANGLE, SET_ANGLES, SET_PULSE, or CALIBRATION_JOG."
    )


class IRISRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        client_ip = self.client_address[0]
        if client_ip != ALLOWED_PC_IP:
            self.wfile.write((json.dumps(response("rejected", error="Client IP not allowed")) + "\n").encode())
            return

        for raw_line in self.rfile:
            if len(raw_line) > MAX_LINE_BYTES:
                self.wfile.write((json.dumps(response("rejected", error="Request too large")) + "\n").encode())
                return
            try:
                request = json.loads(raw_line.decode("utf-8"))
                supplied_token = str(request.pop("token", ""))
                if not hmac.compare_digest(supplied_token, TOKEN):
                    result = response("rejected", error="Invalid token")
                else:
                    result = handle_command(request)
            except Exception as exc:
                result = response("rejected", error=str(exc))
            self.wfile.write((json.dumps(result) + "\n").encode())
            self.wfile.flush()


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    print(f"IRIS Pi controller listening on {HOST}:{PORT}")
    print(f"Only PC {ALLOWED_PC_IP} is allowed to connect.")
    try:
        with ThreadedTCPServer((HOST, PORT), IRISRequestHandler) as server:
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nIRIS Pi controller stopping.")
    finally:
        if pca is not None:
            pca.deinit()
