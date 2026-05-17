# Hardware Notes

## Main Components

- ESP32 WROOM-32D
- PCA9685 16-channel PWM driver
- 6-DOF 3D-printed robotic arm
- Hobby servos
- USB camera
- HP PS-6241-4HP ATX PSU, 5V / 17A rail
- ChArUco calibration board

## Wiring Notes

The PCA9685 receives I2C commands from the ESP32:

```text
ESP32 GPIO21 -> PCA9685 SDA
ESP32 GPIO22 -> PCA9685 SCL
```

Servo power is supplied separately through the PCA9685 power terminal.

The most important electrical rule:

```text
ESP32 GND, PCA9685 GND, and PSU GND must be common.
```

## Mechanical Notes

The arm is based on MakerWorld design #1134925 by Emre Kalem and modified for IRIS.

Custom changes include:

- enlarged gripper opening;
- base wire-protection support;
- 608 and 6203 bearing integration;
- custom PSU enclosure, with the OpenSCAD source in `hardware/hp_psu_enclosure.scad`.

## Power

The recovered HP ATX PSU provides enough current headroom for servo spikes. Small 5V adapters may brown out when several servos move at the same time.
