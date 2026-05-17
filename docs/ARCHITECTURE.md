# IRIS Architecture

IRIS is organized as layered robotics software. Each layer has a clear responsibility, so the AI planner does not need to know about PWM pulses, and the ESP32 does not need to understand high-level tasks.

## Layer 3 - Voice & Hearing

The live interface provides the human-facing loop:

- listens for commands;
- responds conversationally;
- forwards task requests to the robotic execution layer;
- displays camera feedback and action traces.

The voice layer should not directly control servos. It delegates to structured tools exposed by the bridge.

## Layer 2 - Brain

The brain layer is the embodied reasoning/orchestration layer.

It receives a command such as:

```text
pick up the 3x3 cube
```

Then it plans a sequence:

1. detect the object;
2. open the gripper;
3. move above the object;
4. align wrist roll;
5. descend;
6. visually check proximity;
7. close the gripper;
8. lift and verify.

The model does not send raw PWM or servo values. It calls structured functions such as `detect_objects`, `move_arm`, `open_gripper`, and `close_gripper`.

## Layer 2 - Eyes

The vision layer converts camera pixels to robot coordinates.

Recommended baseline:

```text
camera calibration -> ChArUco board solvePnP -> camera-to-robot transform -> ray-plane intersection
```

This avoids relying on a fragile manual correction map as the default path.

## Layer 2 - Cerebellum

The cerebellum layer solves inverse kinematics.

Input:

```text
x, y, z target in millimeters
```

Output:

```text
servo angles for base, shoulder, elbow, wrist pitch, wrist roll
```

The URDF model in `files/iris_arm.urdf` defines the kinematic chain used by IKPy.

## Layer 1 - Spine

The spine is the low-level control path:

- Python bridge computes target servo angles.
- ESP32 receives commands over serial.
- PCA9685 generates stable 50 Hz PWM.
- Servos execute the movement.

## Layer 0 - Body

The physical arm is a modified 3D-printed 6-DOF design with:

- larger custom gripper;
- bearing-supported joints;
- wire protection in the base;
- dedicated ATX PSU enclosure.

Mechanical flex, servo backlash, and gripper geometry are expected sources of error and should be treated as part of the system.
