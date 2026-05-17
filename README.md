# IRIS Vision Robot Arm

IRIS, short for **Intelligent Robotic Interactive System**, is an open-source 6-DOF robotic arm project that connects real-time AI reasoning, camera-based workspace calibration, inverse kinematics, and an ESP32/PCA9685 servo control stack.

The goal is simple: let an AI system see objects on a table, understand natural language commands, plan a manipulation sequence, and move the physical arm without hard-coded task routines.

## What IRIS Does

- Understands natural language tasks through a voice/live interface.
- Uses Gemini Robotics-style embodied reasoning as the high-level planner.
- Detects objects in the camera feed and converts image pixels into robot workspace coordinates.
- Solves inverse kinematics with IKPy for a 6-DOF arm.
- Sends smooth servo targets to an ESP32 through a serial bridge.
- Drives hobby servos through a PCA9685 PWM controller.
- Uses a board-only ChArUco calibration workflow as the recommended stable baseline.

## Architecture

```text
Layer 3 - Voice & Hearing
  Web live UI, microphone/speech flow, Romanian conversation

Layer 2 - Brain
  Gemini Robotics / embodied reasoning orchestration through tool calls

Layer 2 - Eyes
  Camera calibration, ChArUco board detection, pixel-to-robot coordinates

Layer 2 - Cerebellum
  IKPy inverse kinematics, URDF model, servo angle planning

Layer 1 - Spine
  ESP32 + serial protocol + PCA9685 PWM control

Layer 0 - Body
  3D-printed 6-DOF arm, custom gripper, bearings, wire protection, PSU enclosure
```

Read more in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository Layout

```text
files/iris_bridge.py       Main HTTP bridge: AI tools, IK, camera, serial control
files/iris_live.html       Live voice/control UI
files/iris_visualizer.html Browser-based arm visualizer and manual controls
files/iris_vision_v3.py    Board-only solvePnP/ray-plane camera calibration
files/calibrate_camera.py  Camera intrinsics calibration
files/iris_arm.urdf        Robot arm kinematic model for IKPy
files/iris_gamepad.py      Optional gamepad/manual control
hardware/                 Hardware notes and printable/custom parts
docs/                     Setup, calibration, and architecture documentation
```

## Hardware Summary

- 6-DOF 3D-printed robotic arm based on MakerWorld design #1134925 by Emre Kalem, with custom modifications.
- Enlarged gripper opening for larger demo objects.
- Wire protection inside the base to avoid cable damage around rotating gears.
- 608 and 6203 bearings in selected joints for smoother movement.
- ESP32 WROOM-32D as the low-level controller.
- PCA9685 16-channel PWM driver at 50 Hz.
- Recovered HP PS-6241-4HP ATX PSU, 5V / 17A rail.
- Common ground between PSU, PCA9685, and ESP32 is mandatory.

## Software Requirements

- Python 3.11+
- OpenCV with ArUco/ChArUco support
- IKPy
- NumPy / SciPy
- PySerial
- A Gemini API key for AI orchestration

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

## Quick Start

1. Calibrate the camera intrinsics:

   ```bash
   python files/calibrate_camera.py
   ```

2. Run board-only workspace calibration:

   ```bash
   python files/iris_vision_v3.py
   ```

   Show the ChArUco board fully, wait for enough detected corners, press `SPACE`, then `S`.

3. Start the bridge:

   ```bash
   python files/iris_bridge.py
   ```

4. Open the live UI:

   ```text
   http://localhost:8765/live
   ```

## Calibration Philosophy

The public baseline intentionally uses **board-only calibration**. Experimental ID21 gripper-marker correction and manual residual maps are not part of the recommended baseline because they can overfit or amplify physical setup errors.

The stable path is:

```text
camera intrinsics -> ChArUco board pose -> camera-to-robot transform -> ray-plane intersection
```

See [docs/CALIBRATION.md](docs/CALIBRATION.md).

## Project Status

IRIS is an active prototype. It works as a real AI-controlled robotic arm, but precision depends heavily on:

- camera mounting rigidity;
- ChArUco board placement and offset;
- servo backlash and mechanical flex;
- gripper geometry;
- object shape and lighting.

The current recommended baseline is designed to be understandable and rebuildable before adding more complex correction layers.

## License

License not selected yet. Add a `LICENSE` file before public release.

## Credits

Created by Mirauta Alexandru and Cardas Codrin.

The mechanical arm is based on MakerWorld design #1134925 by Emre Kalem, with custom modifications for the IRIS project.
