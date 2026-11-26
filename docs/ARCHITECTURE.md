# IRIS Architecture

IRIS uses a two-computer architecture. The Dell is the brain; the Raspberry Pi is the real-time hardware controller.

## Dell Precision 3460 — Brain

The Dell runs:

- Logitech C270 camera capture;
- OpenCV and object detection;
- ChArUco camera/workspace calibration;
- pixel-to-robot coordinate conversion;
- AI planning and the browser/voice interface;
- IKPy inverse kinematics and URDF calculations;
- the HTTP bridge on port 8765.

The existing vision and IK pipeline remains on the Dell. The Raspberry Pi is not expected to run camera vision or AI models.

## Network Link

The Dell bridge sends JSON-line commands to the Raspberry Pi over the trusted local network.

- Dell address: `192.168.1.213`
- Raspberry Pi address: `192.168.1.196`
- Pi controller port: `8766`
- Both devices must have the same `IRIS_TOKEN`.
- The Pi accepts commands only from the configured `IRIS_ALLOWED_PC_IP`.

The token is sent over plain TCP. Keep this service on a trusted LAN and do not expose port 8766 to the internet.

## Raspberry Pi 4 — Hardware Controller

The Pi runs `pi/iris_pca9685_server.py` and controls the PCA9685 over I2C.

| Pi connection | PCA9685 |
| --- | --- |
| Physical pin 1 / 3.3V | VCC |
| Physical pin 3 / GPIO2 / SDA | SDA |
| Physical pin 5 / GPIO3 / SCL | SCL |
| Physical pin 6 / GND | GND |

The external regulated 5V supply powers the servo V+ rail. The Pi must not power the servos. Pi GND, PCA9685 GND, and power-supply GND must be common.

## PCA9685 Channel Map

| Channel | Joint | Servo |
| --- | --- | --- |
| CH0 | Gripper | MG90S |
| CH1 | Wrist rotation | MG90S |
| CH2 | Wrist pitch | MG90S |
| CH3 | Elbow | MG996R |
| CH4 | Shoulder A | MG996R |
| CH5 | Shoulder B | MG996R |
| CH6 | Base | MG996R |

Gripper convention at the public API: **0° = CLOSED; 90° = OPEN**.

## Servo Calibration and Safety Gates

The Pi converts joint angles into pulse widths using `pi/servo_calibration.json`. Every channel is disabled in the committed configuration.

Coordinated angle motion is allowed only when all three conditions are true:

1. The Pi process has `IRIS_ENABLE_HARDWARE=1`.
2. Every channel CH0–CH6 has `"enabled": true` and valid calibrated angle and pulse endpoints in `servo_calibration.json`.
3. `IRIS_ALLOW_COORDINATED_MOTION=1` is explicitly set after calibration and motion-path review. This extra gate defaults to OFF even when the hardware gate is enabled.

A Dell-local browser panel at `http://localhost:8765/calibration` sends the separate `CALIBRATION_JOG` command, which can move exactly one channel inside its temporary `jog_min_us` / `jog_max_us` window. The page and its POST endpoints are restricted to the Dell PC's local addresses. It requires BOTH `IRIS_ENABLE_HARDWARE=1` and `IRIS_CALIBRATION_MODE=1`. The temporary jog window is only ±25 µs around 1500 µs (CH1 wrist rotation: ±25 µs around the user's 1550 µs reference). These are deliberately narrow commissioning limits, not final calibrated travel limits. Keep the servo horn/linkage disconnected and the arm unloaded during this test. Never jog a servo that is buzzing, stalled, hot, or mechanically constrained.

The user's earlier CH1 test used a provisional 1500–1600 µs window around a 1550 µs / 90° reference. That range is NOT installed as a final travel limit. The current calibration-only jog window is narrower (1525–1575 µs), and CH1 remains disabled until its final mechanical limits are measured.

Do not copy old V7 offsets or broad degree limits into the Pi calibration file. Calibrate each joint individually, with the linkage unloaded where possible. Stop immediately if a servo buzzes, stalls, heats, or needs manual force.

Preset motions (hello wave, dance, pattern playback, and PS4 gamepad control) are disabled in the Dell bridge unless `IRIS_ALLOW_PRESET_MOTIONS=1` is explicitly set after calibration and review. Autonomous pick and Live robotics tasks are separately disabled unless `IRIS_ALLOW_AUTONOMOUS_MOTION=1` is explicitly set. Both flags default to OFF.

The network `STOP` command latches the command path and rejects later movement commands until the Pi controller process is deliberately restarted. It preserves the last PWM setpoints so a gravity-loaded arm does not suddenly lose holding torque. This is not a safety-rated emergency stop; the physical servo-power switch remains the emergency power disconnect.

## Deployment Templates

- `pi/iris.env.example` is the Pi environment template. Copy it to `/etc/iris/iris.env`, set a long random `IRIS_TOKEN`, and keep all three hardware/motion flags at `0` during setup.
- `files/iris.env.example` is the Dell environment template. It must use the same token and the known PC/Pi LAN addresses.
- `pi/iris-pca9685.service` is an optional systemd unit for the Pi account `genos`. It expects the project at `/home/genos/IRIS-vision-robot-arm`, the project virtual environment at `.venv`, and the Pi environment file at `/etc/iris/iris.env`.
- The Pi service requires I2C access for its service user. The `i2c` supplementary group is specified in the unit.
- Do not commit real tokens or copy a populated environment file into the repository.
## Motion Transport

The Dell smooth-movement worker sends each interpolation frame as one `SET_ANGLES` JSON-line request rather than opening a TCP request for every servo. The Pi validates the complete channel/angle set and computes every PWM duty cycle before writing any PCA9685 channel. If one requested joint is outside its calibrated limits, the whole frame is rejected before PWM changes.

`SET_ANGLES` is a coordinated-motion command. It requires the separate `IRIS_ALLOW_COORDINATED_MOTION=1` gate and valid calibration for all seven arm channels. Dry-run mode accepts the same request shape for software-path testing but never initializes I2C or writes PWM.

## Command Flow

```text
Camera (Logitech C270)
        |
        v
Dell: OpenCV + object detection + calibration
        |
        v
Dell: AI planner + IKPy + HTTP bridge (:8765)
        |
        | authenticated JSON over TCP (:8766)
        v
Raspberry Pi 4: PCA9685 controller
        |
        | I2C
        v
PCA9685 at 50 Hz
        |
        v
MG90S / MG996R servos
```

## Current Implementation Status

- Dell vision and IK remain in the existing project.
- Dell-to-Pi authenticated TCP client is implemented, including batched movement frames.
- Pi PCA9685 server is implemented with all-or-nothing target validation.
- GitHub Actions syntax checks and mocked Pi safety tests pass.
- All servo channels are disabled by default.
- Hardware movement remains off until per-channel pulse calibration is complete.
- Preset motions and autonomous pick are disabled by default behind separate opt-in flags.
- Cancel latches the Pi command path and holds the last PWM setpoints rather than sending an uncalibrated home pose.
