# Calibration Guide

This repository uses a board-only calibration baseline.

The goal is to make the simplest path reliable before adding correction layers.

## 1. Camera Intrinsics

Run:

```bash
python files/calibrate_camera.py
```

This generates local camera calibration files. They are ignored by Git because they depend on your exact camera and lens.

## 2. Workspace Calibration

Run:

```bash
python files/iris_vision_v3.py
```

Recommended workflow:

1. Place the ChArUco board flat on the robot workspace.
2. Make sure the board is fully visible and stable.
3. Wait until enough ChArUco corners are detected.
4. Press `O` if the board offset from the robot base needs adjustment.
5. Press `SPACE` to compute the camera-to-robot transform.
6. Press `S` to save.
7. Restart or reload `files/iris_bridge.py`.

## Board Offset

The board offset defines where the board origin is in robot coordinates.

If the grid appears shifted consistently, fix the board offset first instead of adding a residual correction map.

## Why Board-Only?

Earlier experiments used an ArUco marker on the gripper and manual correction points. Those can be useful for research, but they can also amplify noise from:

- marker flex;
- gripper tip uncertainty;
- servo backlash;
- camera pose drift;
- local overfitting.

For a public baseline, the recommended path is:

```text
ChArUco board -> solvePnP -> ray-plane intersection
```

## Validation

After calibration:

- click known positions in `iris_vision_v3.py`;
- compare the reported robot coordinates against the expected board/grid coordinates;
- test a few pick targets near the center and near the workspace edges;
- avoid changing detection offsets until the board calibration is stable.
