#!/usr/bin/env python3
"""
IRIS 6-DOF Robot Arm — IK with ikpy + URDF
Includes servo offset mapping (IK radians <-> servo degrees)
"""

import numpy as np
import ikpy.chain
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1. Load chain from URDF
# ============================================================
iris_chain = ikpy.chain.Chain.from_urdf_file(
    "/home/claude/iris/iris_arm.urdf",
    active_links_mask=[False, True, True, True, True, True, False, False]
    # [origin, base_rot, shoulder, elbow, wrist_pitch, wrist_roll, gripper(fixed for IK), ee]
)

print("=== IRIS IK Chain ===")
print(f"Links: {len(iris_chain.links)}")
for i, link in enumerate(iris_chain.links):
    print(f"  [{i}] {link.name}")
print()

# ============================================================
# 2. Servo offset mapping
# ============================================================
# Servo offsets: the servo degree value when robot is at IK "zero" (horizontal)
SERVO_OFFSETS = {
    'base_rotation': 90,    # CH6: 90° = center
    'shoulder':      150,   # CH4: 150° = horizontal
    'elbow':         50,    # CH3: 50° = straight
    'wrist_pitch':   90,    # CH2: 90° = straight
    'wrist_roll':    55,    # CH1: 55° = straight
}

# Direction multipliers (how servo degrees relate to IK positive rotation)
# +1 = servo increase = IK positive rotation
# -1 = servo increase = IK negative rotation
SERVO_DIRECTIONS = {
    'base_rotation': 1,     # CH6: creste = roteste in sensul pozitiv
    'shoulder':      -1,    # CH4: scade = ridica (IK pozitiv = ridica)
    'elbow':         -1,    # CH3: scade = drepteaza
    'wrist_pitch':   -1,    # CH2: scade = urca
    'wrist_roll':    1,     # CH1: standard
}

SERVO_LIMITS = {
    'base_rotation': (0, 180),
    'shoulder':      (0, 150),
    'elbow':         (20, 180),
    'wrist_pitch':   (0, 170),
    'wrist_roll':    (0, 180),
}

JOINT_NAMES = ['base_rotation', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll']


def ik_to_servo(ik_angles_rad):
    """Convert IK joint angles (radians) to servo degrees."""
    servo_angles = {}
    # ik_angles_rad includes all links (base_link, joints, ee)
    # Active joints are at indices 1-5
    for i, name in enumerate(JOINT_NAMES):
        ik_rad = ik_angles_rad[i + 1]  # +1 because index 0 is base_link (fixed)
        ik_deg = np.degrees(ik_rad)
        servo_deg = SERVO_OFFSETS[name] + SERVO_DIRECTIONS[name] * ik_deg
        # Clamp to servo limits
        lo, hi = SERVO_LIMITS[name]
        servo_deg = np.clip(servo_deg, lo, hi)
        servo_angles[name] = round(servo_deg, 1)

    # Mirror for CH5
    servo_angles['shoulder_B'] = round(180 - servo_angles['shoulder'], 1)
    return servo_angles


def servo_to_ik(servo_dict):
    """Convert servo degrees to IK joint angles (radians). Returns full array."""
    ik_angles = np.zeros(len(iris_chain.links))
    for i, name in enumerate(JOINT_NAMES):
        servo_deg = servo_dict.get(name, SERVO_OFFSETS[name])
        ik_deg = (servo_deg - SERVO_OFFSETS[name]) * SERVO_DIRECTIONS[name]
        ik_angles[i + 1] = np.radians(ik_deg)
    return ik_angles


# ============================================================
# 3. Test: Forward Kinematics at home position
# ============================================================
home_ik = np.zeros(len(iris_chain.links))  # All IK angles = 0 = home
fk_home = iris_chain.forward_kinematics(home_ik)

print("=== Home Position (FK) ===")
print(f"End effector position: x={fk_home[0,3]*1000:.1f}mm, y={fk_home[1,3]*1000:.1f}mm, z={fk_home[2,3]*1000:.1f}mm")
print(f"Servo angles at home: {ik_to_servo(home_ik)}")
print()

# ============================================================
# 4. Test: IK to a target position
# ============================================================
# Target: 300mm in front, 150mm up from base
target_pos = [0.30, 0.0, 0.20]
print(f"=== IK Test: Target ({target_pos[0]*1000:.0f}, {target_pos[1]*1000:.0f}, {target_pos[2]*1000:.0f}) mm ===")

ik_result = iris_chain.inverse_kinematics(target_pos)
fk_result = iris_chain.forward_kinematics(ik_result)
actual_pos = fk_result[:3, 3]
error = np.linalg.norm(np.array(target_pos) - actual_pos) * 1000

print(f"IK solution (rad): {[round(a,4) for a in ik_result]}")
print(f"FK verification:   x={actual_pos[0]*1000:.1f}mm, y={actual_pos[1]*1000:.1f}mm, z={actual_pos[2]*1000:.1f}mm")
print(f"Error: {error:.2f} mm")
print(f"Servo angles: {ik_to_servo(ik_result)}")
print()

# ============================================================
# 5. Test multiple positions
# ============================================================
test_targets = [
    [0.35, 0.0, 0.15],   # In front, slightly up
    [0.20, 0.15, 0.20],  # Front-left, up
    [0.25, -0.10, 0.10], # Front-right, low
    [0.15, 0.0, 0.35],   # Close, high up
    [0.40, 0.0, 0.00],   # Far forward, base height
]

print("=== Multi-target IK Test ===")
for target in test_targets:
    ik_res = iris_chain.inverse_kinematics(target)
    fk_res = iris_chain.forward_kinematics(ik_res)
    pos = fk_res[:3, 3]
    err = np.linalg.norm(np.array(target) - pos) * 1000
    servos = ik_to_servo(ik_res)
    
    # Check if servos are within limits
    in_limits = all(
        SERVO_LIMITS[name][0] <= servos[name] <= SERVO_LIMITS[name][1]
        for name in JOINT_NAMES
    )
    
    print(f"Target: ({target[0]*1000:6.0f}, {target[1]*1000:6.0f}, {target[2]*1000:6.0f})mm "
          f"-> Actual: ({pos[0]*1000:6.1f}, {pos[1]*1000:6.1f}, {pos[2]*1000:6.1f})mm "
          f"| Err: {err:5.2f}mm "
          f"| {'OK' if in_limits else 'LIMIT!'}")
print()

# ============================================================
# 6. Visualization
# ============================================================
fig = plt.figure(figsize=(16, 10))

# Plot 1: Home position
ax1 = fig.add_subplot(121, projection='3d')
iris_chain.plot(home_ik, ax1, target=None)
ax1.set_title("IRIS - Home Position (Zero)")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.set_xlim(-0.3, 0.6)
ax1.set_ylim(-0.3, 0.3)
ax1.set_zlim(-0.1, 0.5)

# Plot 2: IK solution
ax2 = fig.add_subplot(122, projection='3d')
iris_chain.plot(ik_result, ax2, target=target_pos)
ax2.set_title(f"IRIS - IK to ({target_pos[0]*1000:.0f}, {target_pos[1]*1000:.0f}, {target_pos[2]*1000:.0f})mm")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_zlabel("Z (m)")
ax2.set_xlim(-0.3, 0.6)
ax2.set_ylim(-0.3, 0.3)
ax2.set_zlim(-0.1, 0.5)

plt.tight_layout()
plt.savefig("/home/claude/iris/iris_ik_test.png", dpi=150, bbox_inches='tight')
print("Plot saved to iris_ik_test.png")

# Also plot all test targets
fig2 = plt.figure(figsize=(10, 8))
ax3 = fig2.add_subplot(111, projection='3d')
for target in test_targets:
    ik_res = iris_chain.inverse_kinematics(target)
    iris_chain.plot(ik_res, ax3, target=target)
ax3.set_title("IRIS - Multiple IK Solutions")
ax3.set_xlabel("X (m)")
ax3.set_ylabel("Y (m)")
ax3.set_zlabel("Z (m)")
ax3.set_xlim(-0.3, 0.6)
ax3.set_ylim(-0.3, 0.3)
ax3.set_zlim(-0.1, 0.5)
plt.tight_layout()
plt.savefig("/home/claude/iris/iris_multi_targets.png", dpi=150, bbox_inches='tight')
print("Multi-target plot saved to iris_multi_targets.png")
