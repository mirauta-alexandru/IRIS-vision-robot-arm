# IRIS Vision Robot Arm

[Versiunea în română](README.ro.md)

![IRIS logo](assets/iris-logo-banner.png)

**IRIS** stands for **Intelligent Robotic Interactive System**: an open-source, AI-powered 6-DOF robotic arm built to see objects on a table, understand natural language commands, plan manipulation steps, and move a physical arm through computer vision, calibration, inverse kinematics, and ESP32/PCA9685 servo control.

IRIS is not designed as a robot that only replays pre-programmed motions. The goal is to let an AI system receive a new command, inspect the scene in real time, decide what needs to happen, and control the arm through structured functions.

[Project presentation](docs/prezentare-iris.pdf)

## Version 7 Update

This release focuses on the current IRIS V7 control stack:

- improved Gemini Live tools for direct robot gestures such as greeting and dancing;
- a 30-second local MP3 dance routine with beat-based speed phases. Place your local audio at `files/media/iris_dance.mp3`;
- safer shoulder limits, with CH4 constrained to 30-160 degrees and CH5 mirrored from CH4;
- a fresh PS4 manual controller script for direct servo testing through the bridge;
- improved visualizer controls for PS4 mode, hello wave, and music dance routines.

## Build Overview

To build IRIS from scratch, follow the project in this order:

1. Print and assemble the robotic arm.
2. Install the electronics: ESP32, PCA9685, servos, power supply, and common ground.
3. Connect a fixed overhead camera.
4. Install the Python software.
5. Calibrate the camera intrinsics.
6. Calibrate the robot workspace with the ChArUco board.
7. Start the bridge and open the live interface.
8. Test manual movement before running autonomous AI tasks.

The 3D-printable parts and hardware files belong in the [hardware](hardware) folder. The current repository already includes hardware notes and the custom PSU enclosure source; the full printable arm files will be added there later.

Assembly reference video:
[IRIS / 6-DOF arm assembly video](https://www.youtube.com/watch?v=CHV36hu9z3E)

## Hardware Needed

Main parts:

- 3D-printed 6-DOF robotic arm parts;
- ESP32 WROOM-32D;
- PCA9685 16-channel PWM servo driver;
- hobby servos for the arm joints and gripper;
- high-current 5V power supply for the servos;
- USB or phone camera mounted rigidly above the workspace;
- ChArUco calibration board, printed flat and mounted on a rigid surface;
- jumper wires, servo extensions, screws, bearings, and mechanical fasteners.

Important electrical notes:

- Power the servos from the external 5V supply, not from the ESP32.
- Connect ESP32 GND, PCA9685 GND, and power-supply GND together.
- ESP32 talks to the PCA9685 through I2C: `GPIO21 -> SDA`, `GPIO22 -> SCL`.
- Test one servo at a time before moving the full arm.

## Run the Software

Clone the repository:

```bash
git clone https://github.com/mirauta-alexandru/IRIS-vision-robot-arm.git
cd IRIS-vision-robot-arm
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

Calibrate the camera:

```bash
python files/calibrate_camera.py
```

Calibrate the robot workspace:

```bash
python files/iris_vision_v3.py
```

Start the bridge:

```bash
python files/iris_bridge.py
```

Open the live interface:

```text
http://localhost:8765/live
```

Optional manual gamepad control:

```bash
python files/iris_ps4_manual.py
```

## What IRIS Is

IRIS is an autonomous robotics prototype made from a 3D-printed 6-DOF arm, an overhead camera, an AI reasoning layer, ChArUco-based workspace calibration, an IKPy inverse-kinematics stack, and an ESP32/PCA9685 low-level control path.

In practical terms, IRIS can:

- listen and respond through a real-time live interface;
- see the workspace through a camera;
- detect objects and convert image pixels into robot coordinates;
- plan manipulation steps;
- solve inverse kinematics with IKPy;
- send joint targets to an ESP32;
- drive hobby servos through a PCA9685 PWM module;
- be extended with new robot functions without rewriting the whole system.

## Architecture

```text
Layer 3 - Voice and Hearing
  Live interface, microphone, spoken responses, natural-language commands

Layer 2 - Brain
  Gemini Robotics-style embodied reasoning, planning, and function calling

Layer 2 - Eyes
  Camera feed, ChArUco calibration, pixel-to-robot coordinate conversion

Layer 2 - Cerebellum
  IKPy, URDF model, inverse kinematics, servo target generation

Layer 1 - Spine
  ESP32, serial protocol, PCA9685, stable PWM output for servos

Layer 0 - Body
  3D-printed 6-DOF robotic arm, modified gripper, bearings, ATX power supply
```

Technical notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## How It Works

1. The user gives a natural command, such as "pick up the green cube".
2. The camera sends a frame to the vision layer.
3. The system detects the target object and estimates its position on the table.
4. ChArUco calibration converts that image position into robot workspace coordinates.
5. The AI planner breaks the task into steps: approach, align, descend, grasp, lift.
6. IKPy converts the XYZ target into joint angles.
7. The ESP32 receives commands and drives the servos through the PCA9685 module.
8. The camera can be used again to verify the result and make small corrections.

## Repository Layout

```text
files/iris_bridge.py       Main bridge: AI tools, IK, camera, HTTP, serial control
files/iris_live.html       Real-time voice and control interface
files/iris_visualizer.html Browser visualizer for camera, arm state, and commands
files/iris_vision_v3.py    ChArUco workspace calibration
files/calibrate_camera.py  Camera intrinsics calibration
files/iris_arm.urdf        Kinematic model used by IKPy
files/iris_ps4_manual.py   Optional PS4 manual servo control
files/media/              Local, git-ignored demo audio folder
hardware/                  Hardware notes and custom printable parts
docs/                      Architecture, setup, calibration, and project deck
assets/                    Repository images and social preview assets
```

## Hardware Details

IRIS uses a 6-DOF 3D-printed robotic arm based on MakerWorld design #1134925 by Emre Kalem, with several custom modifications:

- enlarged gripper opening for larger demo objects;
- wire-protection support inside the base;
- 608 and 6203 bearings in selected joints for smoother movement;
- custom enclosure for a recovered HP PS-6241-4HP ATX power supply;
- 5V / 17A servo power rail from the recovered PSU;
- ESP32 WROOM-32D as the low-level controller;
- PCA9685 16-channel PWM module at 50 Hz.

Important electrical rule: **ESP32 GND, PCA9685 GND, and PSU GND must be common**.

## Software Requirements

Main requirements:

- Python 3.11+
- OpenCV with ArUco/ChArUco support
- IKPy
- NumPy / SciPy
- PySerial
- Requests
- Pygame for optional gamepad control
- a Gemini API key for the AI layer

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

## Quick Start Checklist

1. Calibrate the camera intrinsics:

   ```bash
   python files/calibrate_camera.py
   ```

2. Calibrate the robot workspace with the ChArUco board:

   ```bash
   python files/iris_vision_v3.py
   ```

   Place the board on the table, make sure it is fully visible, wait for enough detected corners, then press `SPACE` and `S`.

3. Start the bridge:

   ```bash
   python files/iris_bridge.py
   ```

4. Open the live interface:

   ```text
   http://localhost:8765/live
   ```

## Calibration

The public baseline uses a simple and stable calibration flow:

```text
camera intrinsics -> ChArUco solvePnP -> camera-to-robot transform -> ray-plane intersection
```

Earlier experiments used gripper-marker correction and manual residual maps. They are useful for research, but they can also amplify mechanical issues such as backlash, flex, marker movement, imperfect measurements, or local offsets. For the public baseline, board-only calibration is easier to understand, reproduce, and debug.

Full guide: [docs/CALIBRATION.md](docs/CALIBRATION.md)

## Why IRIS Is Different

IRIS is a step toward robots that do more than replay programmed motions. It sees, listens, reasons, speaks, and acts. The AI layer does not merely describe what should happen; it controls a physical system that tries to do it in the real world.

Most AI still lives behind screens: text, images, code, conversations. IRIS explores what happens when that intelligence gets a body: an object on a table, a spoken command, a planned trajectory, and a real movement.

## Project Status

IRIS is an active prototype. It works as a real AI-controlled robotic arm, but precision depends heavily on:

- camera rigidity;
- ChArUco calibration quality;
- physical board placement;
- servo backlash;
- mechanical flex;
- gripper geometry;
- lighting, object shape, and background.

The current goal is a clean, understandable open-source baseline before adding more complex correction or training layers.

## Roadmap

Possible IRIS V2 directions:

- higher-precision mechanics with better actuators than hobby servos;
- finer manipulation for fragile objects;
- simulation-to-real-world training in Isaac Sim or MuJoCo;
- a dedicated VLA model trained on motion policies;
- custom datasets for detection and manipulation.

## Open Source

This project is public for people who want to learn, build, modify, and push forward the idea of an accessible AI robot.

If you have a 3D printer, some electronics, patience for calibration, and curiosity, this repository is a starting point for building your own version of IRIS.

## License

No license has been selected yet. Add a `LICENSE` file before an official public release.

## Credits

Created by **Mirăuță Alexandru** and **Cardaș Codrin**.

The mechanical arm is based on MakerWorld design #1134925 by **Emre Kalem**, modified for the IRIS project.
