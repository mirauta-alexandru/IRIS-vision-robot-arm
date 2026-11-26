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

Coordinated angle motion is allowed only when all seven channels have valid calibrated pulse limits and BOTH conditions are true:

1. The Pi process has `IRIS_ENABLE_HARDWARE=1`.
2. Every channel CH0–CH6 has `"enabled": true` and valid calibrated pulse endpoints in `servo_calibration.json`.

A separate `CALIBRATION_JOG` command can move exactly one channel inside its temporary `jog_min_us` / `jog_max_us` window. It requires BOTH `IRIS_ENABLE_HARDWARE=1` and `IRIS_CALIBRATION_MODE=1`. The temporary jog window is only ±25 µs around 1500 µs (CH1 wrist rotation: ±25 µs around the user's 1550 µs reference). These are deliberately narrow commissioning limits, not final calibrated travel limits. Keep the servo horn/linkage disconnected and the arm unloaded during this test. Never jog a servo that is buzzing, stalled, hot, or mechanically constrained.

CH1 contains a provisional 1500–1600 µs range corresponding to 85–95° around the user's 1550 µs / 90° reference, but it remains disabled. It must be mechanically checked before enabling.

Do not copy old V7 offsets or broad degree limits into the Pi calibration file. Calibrate each joint individually, with the linkage unloaded where possible. Stop immediately if a servo buzzes, stalls, heats, or needs manual force.

The network `STOP` command disables PWM outputs when hardware mode is active, but it is not a safety-rated emergency stop. The physical servo-power switch remains the actual emergency power disconnect.

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
- Dell-to-Pi authenticated TCP client is implemented.
- Pi PCA9685 server is implemented.
- All servo channels are disabled by default.
- Hardware movement remains off until per-channel pulse calibration is complete.
- Automatic pick, dance, and broad legacy poses must not be used as calibration tests.
