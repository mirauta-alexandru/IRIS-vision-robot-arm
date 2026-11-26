"""Safety and dry-run tests for the IRIS Pi PCA9685 controller.

The hardware libraries are replaced with tiny fakes so these tests never touch I2C.
Run from the repository root with: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "pi" / "iris_pca9685_server.py"


class FakePCA9685:
    instances = []

    def __init__(self, i2c):
        self.i2c = i2c
        self.frequency = None
        self.channels = [types.SimpleNamespace(duty_cycle=0) for _ in range(16)]
        self.deinitialized = False
        self.__class__.instances.append(self)

    def deinit(self):
        self.deinitialized = True


def load_controller(*, hardware="0", calibration="0", coordinated="0"):
    FakePCA9685.instances.clear()
    board = types.SimpleNamespace(SCL=object(), SDA=object())
    busio = types.SimpleNamespace(I2C=lambda scl, sda: (scl, sda))
    pca_module = types.SimpleNamespace(PCA9685=FakePCA9685)
    modules = {
        "board": board,
        "busio": busio,
        "adafruit_pca9685": pca_module,
    }
    env = {
        "IRIS_TOKEN": "unit-test-token",
        "IRIS_ENABLE_HARDWARE": hardware,
        "IRIS_CALIBRATION_MODE": calibration,
        "IRIS_ALLOW_COORDINATED_MOTION": coordinated,
    }
    with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=False):
        spec = importlib.util.spec_from_file_location(
            f"iris_pi_controller_test_{hardware}_{calibration}_{coordinated}",
            SERVER_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class PiControllerDryRunTests(unittest.TestCase):
    def test_dry_run_never_initializes_pca_and_accepts_test_targets(self):
        controller = load_controller()
        self.assertIsNone(controller.pca)
        self.assertEqual(controller.handle_command({"command": "PING"})["status"], "ok")
        self.assertEqual(
            controller.handle_command(
                {"command": "SET_ANGLE", "channel": 0, "angle": 45}
            )["status"],
            "dry_run",
        )
        self.assertEqual(
            controller.handle_command(
                {"command": "SET_PULSE", "channel": 0, "pulse_us": 1500}
            )["status"],
            "dry_run",
        )
        self.assertEqual(FakePCA9685.instances, [])

    def test_dry_run_cannot_use_calibration_jog(self):
        controller = load_controller()
        with self.assertRaises(PermissionError):
            controller.handle_command(
                {"command": "CALIBRATION_JOG", "channel": 0, "pulse_us": 1500}
            )

    def test_stop_latches_and_blocks_later_targets(self):
        controller = load_controller()
        result = controller.handle_command({"command": "STOP"})
        self.assertTrue(result["stop_latched"])
        with self.assertRaises(PermissionError):
            controller.handle_command(
                {"command": "SET_ANGLE", "channel": 0, "angle": 45}
            )


class PiControllerHardwareGateTests(unittest.TestCase):
    def test_calibration_jog_is_bounded_and_single_channel(self):
        controller = load_controller(hardware="1", calibration="1")
        pca = FakePCA9685.instances[0]
        result = controller.handle_command(
            {"command": "CALIBRATION_JOG", "channel": 1, "pulse_us": 1550}
        )
        self.assertEqual(result["status"], "calibration_jogged")
        self.assertTrue(result["hardware_moved"])
        self.assertNotEqual(pca.channels[1].duty_cycle, 0)
        self.assertEqual(
            sum(ch.duty_cycle != 0 for ch in pca.channels),
            1,
            "A calibration jog must write only the selected PCA channel.",
        )
        with self.assertRaises(ValueError):
            controller.handle_command(
                {"command": "CALIBRATION_JOG", "channel": 1, "pulse_us": 1576}
            )
        with self.assertRaises(ValueError):
            controller.handle_command(
                {"command": "CALIBRATION_JOG", "channel": 7, "pulse_us": 1500}
            )

    def test_coordinated_motion_is_locked_even_when_hardware_is_on(self):
        controller = load_controller(hardware="1", calibration="1")
        with self.assertRaises(PermissionError):
            controller.handle_command(
                {"command": "SET_ANGLE", "channel": 0, "angle": 45}
            )

    def test_batched_targets_validate_every_joint_before_any_pwm_write(self):
        controller = load_controller(hardware="1", coordinated="1")
        pca = FakePCA9685.instances[0]
        for item in controller.CHANNELS.values():
            item["enabled"] = True
            item["angle_min"] = 0
            item["angle_max"] = 180
            item["pulse_min_us"] = 1000
            item["pulse_max_us"] = 2000
        with self.assertRaises(ValueError):
            controller.handle_command({
                "command": "SET_ANGLES",
                "angles": {"0": 45, "1": 90, "2": 181},
            })
        self.assertTrue(all(channel.duty_cycle == 0 for channel in pca.channels))
        result = controller.handle_command({
            "command": "SET_ANGLES",
            "angles": {"0": 45, "1": 90},
        })
        self.assertEqual(result["status"], "moved")
        self.assertTrue(result["hardware_moved"])
        self.assertNotEqual(pca.channels[0].duty_cycle, 0)
        self.assertNotEqual(pca.channels[1].duty_cycle, 0)
        self.assertEqual(pca.channels[2].duty_cycle, 0)

    def test_disabled_channel_cannot_receive_direct_pulse(self):
        controller = load_controller(hardware="1", calibration="1")
        with self.assertRaises(PermissionError):
            controller.handle_command(
                {"command": "SET_PULSE", "channel": 0, "pulse_us": 1500}
            )

    def test_stop_latches_calibration_jog(self):
        controller = load_controller(hardware="1", calibration="1")
        controller.handle_command({"command": "STOP"})
        with self.assertRaises(PermissionError):
            controller.handle_command(
                {"command": "CALIBRATION_JOG", "channel": 0, "pulse_us": 1500}
            )


if __name__ == "__main__":
    unittest.main()
