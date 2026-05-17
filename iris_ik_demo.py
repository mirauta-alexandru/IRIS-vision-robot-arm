#!/usr/bin/env python3
"""
IRIS 6-DOF Robotic Arm — IK Demo & Serial Controller
=====================================================
Folosește ikpy pentru inverse kinematics și trimite unghiuri prin serial la ESP32.

Configurație servo:
  Joint 0: Base rotation      — MG995  (PCA9685 ch0)
  Joint 1: Shoulder           — MG995  (PCA9685 ch1)
  Joint 2: Elbow              — MG995  (PCA9685 ch2)
  Joint 3: Wrist pitch        — MG995  (PCA9685 ch3)
  Joint 4: Wrist roll         — MG90S  (PCA9685 ch4)
  Joint 5: Gripper            — SG90   (PCA9685 ch5)

Dimensiuni segment (estimate din STL, axă-la-axă):
  Base height:    ~70mm  (Alt_Govde Z)
  Lower arm:     ~108mm  (Alt_Kol - lungime principală)
  Upper arm:      ~95mm  (On_Kol - lungime principală)  
  Wrist:          ~65mm  (Bilek + El combined)
  Gripper:        ~55mm  (Parmak length)

Utilizare:
  python iris_ik_demo.py                    # Mod vizualizare (fără serial)
  python iris_ik_demo.py --port /dev/ttyUSB0  # Cu conexiune serial la ESP32
  python iris_ik_demo.py --interactive      # Mod interactiv CLI
"""

import argparse
import sys
import time
import math
import numpy as np

try:
    from ikpy.chain import Chain
    from ikpy.link import OriginLink, URDFLink
except ImportError:
    print("❌ ikpy nu e instalat. Rulează: pip install ikpy")
    sys.exit(1)

# ============================================================
# 1. DEFINIRE LANȚ CINEMATIC (KINEMATIC CHAIN)
# ============================================================
# Dimensiunile sunt estimate din STL bounding boxes.
# ⚠️  IMPORTANT: Măsoară cu șublerul distanțele axă-la-axă
#     pe brațul real și actualizează valorile de mai jos!

# Lungimi segment în METRI (ikpy lucrează în SI)
# Măsurate cu șublerul, axă-la-axă, pe brațul real
BASE_HEIGHT   = 0.075   # 75mm — ch6 (base) → ch4/5 (shoulder)
UPPER_ARM     = 0.152   # 152mm — ch4/5 (shoulder) → ch3 (elbow)
FOREARM       = 0.136   # 136mm — ch3 (elbow) → ch2 (wrist pitch)
WRIST         = 0.030   # 30mm — ch2 (wrist pitch) → ch1 (wrist roll)
GRIPPER_LEN   = 0.038   # 38mm — ch1 (wrist roll) → ch0 (gripper tip)

def solve_ik(chain, target, initial_position=None, max_retries=5):
    """
    Rezolvă IK cu mai multe încercări pentru convergență mai bună.
    Încearcă din seed-uri diferite și alege soluția cu eroarea minimă.
    """
    best_angles = None
    best_error = float('inf')
    
    seeds = [initial_position if initial_position is not None else np.zeros(len(chain.links))]
    
    # Adaugă seed-uri alternative
    for _ in range(max_retries - 1):
        seed = np.zeros(len(chain.links))
        # Randomize doar link-urile active
        for i, active in enumerate(chain.active_links_mask):
            if active:
                link = chain.links[i]
                if hasattr(link, 'bounds') and link.bounds[0] is not None:
                    lo, hi = link.bounds
                    seed[i] = np.random.uniform(lo, hi)
                else:
                    seed[i] = np.random.uniform(-math.pi/2, math.pi/2)
        seeds.append(seed)
    
    for seed in seeds:
        ik_angles = chain.inverse_kinematics(
            target_position=target,
            initial_position=seed,
        )
        fk = chain.forward_kinematics(ik_angles)
        actual = fk[:3, 3]
        error = np.linalg.norm(np.array(target) - actual)
        
        if error < best_error:
            best_error = error
            best_angles = ik_angles
        
        if error < 0.001:  # sub 1mm, suficient de bun
            break
    
    return best_angles, best_error


def build_iris_chain():
    """Construiește lanțul cinematic ikpy pentru IRIS."""
    # Bounds calculate din calibrare fizică:
    #
    # Convenție ikpy: unghi pozitiv pe Y = rotire counter-clockwise văzut din dreapta
    # 
    # Shoulder (ch4): offset=150°, dir=-1
    #   servo 150° = orizontal (IK 0 rad)
    #   servo 60°  = ridicat sus (IK +90° = +π/2 rad, fiindcă dir=-1: 150 + (-1)*90 = 60)
    #   → IK bounds: 0 la +π/2
    #
    # Elbow (ch3): offset=0°, dir=+1
    #   servo 0°   = drept (IK 0 rad)
    #   servo 155° = îndoit în jos (IK -155° = -2.7 rad, negativ = cot coboară)
    #   → servo = 0 + 1 * degrees(ik_rad) → ik_rad negativ dă servo negativ... nu merge!
    #   FIX: direction trebuie să fie -1 pentru elbow!
    #   servo = 0 + (-1) * degrees(ik_rad) → ik_rad=-155° → servo = +155° ✓
    #   → IK bounds: -155° la 0
    #
    # Wrist pitch (ch2): offset=170°, dir=+1  
    #   servo 170° = drept (IK 0 rad)
    #   servo 80°  = wrist sus (IK -90° → 170 + 1*(-90) = 80 ✓)
    #   servo 180° = wrist jos puțin (IK +10° → 170 + 10 = 180 ✓)
    #   → IK bounds: -90° la +10°
    
    chain = Chain(name="IRIS_arm", links=[
        OriginLink(),

        # Base rotation (ch6)
        URDFLink(
            name="base_rotation",
            origin_translation=[0, 0, BASE_HEIGHT],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],
            bounds=(-math.pi / 2, math.pi / 2),
        ),

        # Shoulder (ch4) — 0=orizontal, pozitiv=ridică sus
        URDFLink(
            name="shoulder",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(0, math.radians(90)),  # 0=orizontal, +90=vertical sus
        ),

        # Elbow (ch3) — ch3 crește = cot coboară = unghi IK negativ
        URDFLink(
            name="elbow",
            origin_translation=[UPPER_ARM, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(math.radians(-155), 0),
        ),

        # Wrist pitch (ch2) — 0=drept, negativ=sus
        URDFLink(
            name="wrist_pitch",
            origin_translation=[FOREARM, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(math.radians(-90), math.radians(10)),
        ),

        # Wrist roll (ch1)
        URDFLink(
            name="wrist_roll",
            origin_translation=[WRIST, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[1, 0, 0],
            bounds=(math.radians(-55), math.radians(125)),
        ),

        # End effector
        URDFLink(
            name="end_effector",
            origin_translation=[GRIPPER_LEN, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],
        ),
    ], active_links_mask=[False, True, True, True, True, True, False])

    return chain


# ============================================================
# 2. CONVERSIE UNGHIURI → SERVO GRADE
# ============================================================

# Mapare canale PCA9685 → rol articulație
# CALIBRAT pe robotul real — 28 apr 2026
SERVO_CONFIG = {
    0: {"name": "Gripper",      "role": "gripper",     "type": "SG90",  "offset_deg": 60,  "direction":  1, "min": 0,  "max": 120},
    1: {"name": "Wrist Roll",   "role": "wrist_roll",  "type": "MG90S", "offset_deg": 55,  "direction":  1, "min": 0,  "max": 180},
    2: {"name": "Wrist Pitch",  "role": "wrist_pitch", "type": "MG90S", "offset_deg": 170, "direction":  1, "min": 80, "max": 180},
    3: {"name": "Elbow",        "role": "elbow",       "type": "MG995", "offset_deg": 0,   "direction": -1, "min": 0,  "max": 155},
    4: {"name": "Shoulder A",   "role": "shoulder_a",  "type": "MG995", "offset_deg": 150, "direction": -1, "min": 60, "max": 150},
    5: {"name": "Shoulder B",   "role": "shoulder_b",  "type": "MG995", "offset_deg": 30,  "direction":  1, "min": 30, "max": 120},
    6: {"name": "Base",         "role": "base",        "type": "MG995", "offset_deg": 90,  "direction":  1, "min": 0,  "max": 180},
}

# Mapare: care canale PCA corespund la articulațiile IK
# ikpy produce 5 unghiuri active: [base, shoulder, elbow, wrist_pitch, wrist_roll]
IK_TO_CHANNEL = {
    "base": 6,
    "shoulder": 4,  # ch4 = shoulder_a, ch5 = mirror automat in firmware
    "elbow": 3,
    "wrist_pitch": 2,
    "wrist_roll": 1,
}

def ik_to_servo_angles(ik_angles):
    """
    Convertește unghiurile IK (radiani) în comenzi servo pe canale PCA9685.
    
    ik_angles: array de la ikpy (include link-urile inactive)
    Returns: dict {channel: angle_degrees}
    """
    # ikpy returnează unghiuri pentru TOATE link-urile (7 total)
    # Cele active sunt index 1-5: [base_rot, shoulder, elbow, wrist_pitch, wrist_roll]
    active_angles = ik_angles[1:6]
    ik_names = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll"]
    
    servo_angles = {}
    for ik_name, rad in zip(ik_names, active_angles):
        ch = IK_TO_CHANNEL[ik_name]
        cfg = SERVO_CONFIG[ch]
        deg = cfg["offset_deg"] + cfg["direction"] * math.degrees(rad)
        deg = max(cfg["min"], min(cfg["max"], deg))  # clamp
        servo_angles[ch] = round(deg, 1)
    
    # Gripper la mid-range (nu e controlat de IK)
    grip_cfg = SERVO_CONFIG[0]
    servo_angles[0] = round((grip_cfg["min"] + grip_cfg["max"]) / 2)
    
    return servo_angles


# ============================================================
# 3. COMUNICARE SERIAL CU ESP32
# ============================================================

class SerialController:
    """Trimite comenzi servo prin serial la ESP32 — format IRIS firmware v2."""
    
    def __init__(self, port, baud=115200):
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # Așteaptă ESP32 boot
            # Citește mesajele de boot
            while self.ser.in_waiting:
                self.ser.readline()
            print(f"✅ Conectat la ESP32 pe {port} @ {baud}")
            # Init: firmware-ul pune ch0=120, ch1-6=90 la boot
            self._last_angles = {0: 120, 1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: 90}
        except ImportError:
            print("❌ pyserial nu e instalat. Rulează: pip install pyserial")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Nu pot deschide {port}: {e}")
            sys.exit(1)
    
    def send_angle(self, channel, angle_deg):
        """Trimite o comandă de unghi la un canal PCA9685."""
        # Format firmware: "ch <canal> <grade>\n"
        cmd = f"ch {channel} {angle_deg:.1f}\n"
        self.ser.write(cmd.encode())
        self.ser.flush()
        time.sleep(0.05)
        # Citește răspunsul firmware-ului
        response = ""
        while self.ser.in_waiting:
            response += self.ser.readline().decode(errors='ignore')
        if response.strip():
            for line in response.strip().split('\n'):
                print(f"   ESP32: {line.strip()}")
    
    def send_angle_quiet(self, channel, angle_deg):
        """Trimite o comandă fără a printa răspunsul (pentru interpolare)."""
        cmd = f"ch {channel} {angle_deg:.1f}\n"
        self.ser.write(cmd.encode())
        self.ser.flush()
        # Golește buffer fără a printa
        while self.ser.in_waiting:
            self.ser.readline()
    
    def send_all_angles(self, servo_angles, delay_ms=0.05):
        """Trimite unghiuri la toate servo-urile."""
        for ch, angle in sorted(servo_angles.items()):
            self.send_angle(ch, angle)
            time.sleep(delay_ms)
    
    def send_smooth(self, target_angles, steps=30, duration=1.0):
        """
        Mișcare smooth interpolată de la pozițiile curente la target.
        steps: câți pași intermediari
        duration: durata totală în secunde
        """
        # Citim pozițiile curente din last_angles sau presupunem 90
        if not hasattr(self, '_last_angles'):
            self._last_angles = {ch: SERVO_CONFIG[ch]["offset_deg"] for ch in target_angles}
        
        start = dict(self._last_angles)
        step_delay = duration / steps
        
        for i in range(1, steps + 1):
            t = i / steps
            # Cosine interpolation (smooth ease-in-out)
            t_smooth = (1 - math.cos(t * math.pi)) / 2
            
            for ch, target in target_angles.items():
                start_val = start.get(ch, SERVO_CONFIG[ch]["offset_deg"])
                current = start_val + (target - start_val) * t_smooth
                self.send_angle_quiet(ch, current)
            
            time.sleep(step_delay)
        
        # Update last known positions
        self._last_angles = dict(target_angles)
        
        # Flush remaining serial responses
        time.sleep(0.05)
        while self.ser.in_waiting:
            self.ser.readline()
    
    def send_home(self):
        """Trimite comanda home la firmware."""
        self.ser.write(b"home\n")
        self.ser.flush()
        time.sleep(0.5)
        while self.ser.in_waiting:
            print(f"   ESP32: {self.ser.readline().decode(errors='ignore').strip()}")
    
    def send_grip(self, close=True):
        """Deschide sau închide gripper-ul."""
        cmd = "grip\n" if close else "open\n"
        self.ser.write(cmd.encode())
        self.ser.flush()
        time.sleep(0.1)
        while self.ser.in_waiting:
            print(f"   ESP32: {self.ser.readline().decode(errors='ignore').strip()}")
    
    def send_raw(self, command):
        """Trimite o comandă raw la ESP32."""
        self.ser.write(f"{command}\n".encode())
        self.ser.flush()
        time.sleep(0.1)
        response = ""
        while self.ser.in_waiting:
            response += self.ser.readline().decode(errors='ignore')
        if response.strip():
            for line in response.strip().split('\n'):
                print(f"   ESP32: {line.strip()}")
    
    def close(self):
        self.ser.close()
        print("🔌 Serial deconectat.")


class DummyController:
    """Controller fals pentru testare fără ESP32."""
    
    def send_angle(self, servo_id, angle_deg):
        pass
    
    def send_all_angles(self, servo_angles, delay=0):
        pass
    
    def send_home(self):
        pass
    
    def close(self):
        pass


# ============================================================
# 4. VIZUALIZARE 3D CU MATPLOTLIB
# ============================================================

def plot_arm(chain, angles, target=None, title="IRIS Arm Position"):
    """Afișează brațul 3D cu matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("⚠️  matplotlib nu e instalat. Skip vizualizare.")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Calculează pozițiile fiecărui joint
    frames = chain.forward_kinematics(angles, full_kinematics=True)
    positions = np.array([f[:3, 3] for f in frames])
    
    # Desenează segmentele brațului
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'o-', color='#4A90D9', linewidth=3, markersize=8, label='Arm')
    
    # Marchează joints
    joint_names = ['Origin', 'Base', 'Shoulder', 'Elbow', 'Wrist P', 'Wrist R', 'TCP']
    for i, (pos, name) in enumerate(zip(positions, joint_names)):
        ax.text(pos[0], pos[1], pos[2] + 0.01, name, fontsize=8, ha='center')
    
    # Marchează end effector
    tcp = positions[-1]
    ax.scatter(*tcp, color='red', s=100, zorder=5, label=f'TCP ({tcp[0]*1000:.0f}, {tcp[1]*1000:.0f}, {tcp[2]*1000:.0f})mm')
    
    # Marchează target dacă există
    if target is not None:
        ax.scatter(*target, color='green', s=100, marker='*', zorder=5,
                   label=f'Target ({target[0]*1000:.0f}, {target[1]*1000:.0f}, {target[2]*1000:.0f})mm')
    
    # Setări axe
    arm_reach = UPPER_ARM + FOREARM + WRIST + GRIPPER_LEN
    ax.set_xlim(-arm_reach, arm_reach)
    ax.set_ylim(-arm_reach, arm_reach)
    ax.set_zlim(0, arm_reach + BASE_HEIGHT)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. PRESETURI POZITII
# ============================================================

PRESETS = {
    "home": {
        "description": "Poziție neutră — firmware home command",
        "target_xyz": None,
        "firmware_cmd": "home",
    },
    "forward": {
        "description": "Braț întins înainte",
        "target_xyz": [0.25, 0.0, 0.10],
    },
    "up": {
        "description": "Braț ridicat vertical",
        "target_xyz": [0.05, 0.0, 0.30],
    },
    "pick_low": {
        "description": "Poziție de pickup jos-înainte",
        "target_xyz": [0.20, 0.0, 0.03],
    },
    "left": {
        "description": "Braț întins la stânga",
        "target_xyz": [0.0, 0.20, 0.10],
    },
    "right": {
        "description": "Braț întins la dreapta",
        "target_xyz": [0.0, -0.20, 0.10],
    },
}


# ============================================================
# 6. MOD INTERACTIV
# ============================================================

def interactive_mode(chain, controller):
    """Mod interactiv CLI pentru testare."""
    print("\n" + "="*60)
    print("  IRIS IK Demo — Mod Interactiv")
    print("="*60)
    print("\nComenzi disponibile:")
    print("  xyz <x> <y> <z>   — Mișcă la coordonate XYZ (în mm)")
    print("  preset <name>     — Execută preset (home/forward/up/pick_low/left/right)")
    print("  ch <canal> <deg>  — Control direct canal PCA9685 (0-6, 0-180°)")
    print("  grip / open       — Închide / deschide gripper")
    print("  home              — Home position (firmware)")
    print("  status            — Status ESP32")
    print("  show              — Afișează vizualizare 3D")
    print("  angles            — Afișează unghiurile curente")
    print("  reach             — Afișează workspace-ul")
    print("  raw <cmd>         — Trimite comandă raw la ESP32")
    print("  quit              — Ieșire")
    print()
    
    # Unghiuri curente (start la home)
    current_ik_angles = np.zeros(len(chain.links))
    move_duration = 1.5  # secunde per mișcare
    move_steps = 30
    
    while True:
        try:
            raw = input("IRIS> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not raw:
            continue
        
        parts = raw.split()
        cmd = parts[0].lower()
        
        if cmd == "quit" or cmd == "q":
            print("Bye! 👋")
            break
        
        elif cmd == "speed" and len(parts) >= 2:
            try:
                move_duration = float(parts[1])
                print(f"⏱  Durată mișcare: {move_duration}s")
            except ValueError:
                print("❌ Format: speed <secunde>  (ex: speed 0.5)")
            continue
        
        elif cmd == "home":
            controller.send_home()
            current_ik_angles = np.zeros(len(chain.links))
            controller._last_angles = {0: 120, 1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: 90}
            print("🏠 Home position")
        
        elif cmd == "grip" or cmd == "close":
            controller.send_grip(close=True)
            print("✊ Gripper închis")
        
        elif cmd == "open" or cmd == "release":
            controller.send_grip(close=False)
            print("🖐  Gripper deschis")
        
        elif cmd == "status":
            controller.send_raw("status")
        
        elif cmd == "raw" and len(parts) >= 2:
            raw_cmd = " ".join(parts[1:])
            controller.send_raw(raw_cmd)
        
        elif cmd == "xyz" and len(parts) == 4:
            try:
                x = float(parts[1]) / 1000.0
                y = float(parts[2]) / 1000.0
                z = float(parts[3]) / 1000.0
                target = [x, y, z]
                
                print(f"🎯 Target: ({parts[1]}, {parts[2]}, {parts[3]}) mm")
                
                ik_angles, err_m = solve_ik(chain, target, current_ik_angles)
                
                fk_result = chain.forward_kinematics(ik_angles)
                actual_pos = fk_result[:3, 3]
                error = err_m * 1000
                
                servo_angles = ik_to_servo_angles(ik_angles)
                
                print(f"📐 Channels: " + ", ".join([f"ch{ch}={deg}°" for ch, deg in sorted(servo_angles.items())]))
                print(f"📍 Actual TCP: ({actual_pos[0]*1000:.1f}, {actual_pos[1]*1000:.1f}, {actual_pos[2]*1000:.1f}) mm")
                print(f"📏 Error: {error:.2f} mm")
                
                if error > 10:
                    print(f"⚠️  Eroare mare! Punct posibil în afara workspace-ului.")
                
                controller.send_smooth(servo_angles, steps=move_steps, duration=move_duration)
                current_ik_angles = ik_angles
                print("✅ Trimis la ESP32 (smooth)")
                
            except ValueError:
                print("❌ Format: xyz <x_mm> <y_mm> <z_mm>  (ex: xyz 200 0 100)")
        
        elif cmd == "move" and len(parts) == 4:
            # Safe move: ridică la Z safe → mută XY la înălțime → coboară la Z target
            try:
                x = float(parts[1]) / 1000.0
                y = float(parts[2]) / 1000.0
                z = float(parts[3]) / 1000.0
                safe_z = 0.25  # 250mm — înălțime sigură
                
                print(f"🛡️  Safe move → ({parts[1]}, {parts[2]}, {parts[3]}) mm")
                
                # Pas 1: ridică la Z safe, păstrează XY curent
                fk_current = chain.forward_kinematics(current_ik_angles)
                cur_pos = fk_current[:3, 3]
                
                print(f"   ↑ Ridic la Z={safe_z*1000:.0f}mm...")
                up_target = [cur_pos[0], cur_pos[1], safe_z]
                ik_up, _ = solve_ik(chain, up_target, current_ik_angles)
                servo_up = ik_to_servo_angles(ik_up)
                controller.send_smooth(servo_up, steps=move_steps, duration=move_duration * 0.4)
                current_ik_angles = ik_up
                
                # Pas 2: mută la XY target, menține Z safe
                print(f"   → Mut la X={x*1000:.0f}, Y={y*1000:.0f}...")
                mid_target = [x, y, safe_z]
                ik_mid, _ = solve_ik(chain, mid_target, current_ik_angles)
                servo_mid = ik_to_servo_angles(ik_mid)
                controller.send_smooth(servo_mid, steps=move_steps, duration=move_duration * 0.3)
                current_ik_angles = ik_mid
                
                # Pas 3: coboară la Z target
                print(f"   ↓ Cobor la Z={z*1000:.0f}mm...")
                final_target = [x, y, z]
                ik_final, err_m = solve_ik(chain, final_target, current_ik_angles)
                servo_final = ik_to_servo_angles(ik_final)
                controller.send_smooth(servo_final, steps=move_steps, duration=move_duration * 0.3)
                current_ik_angles = ik_final
                
                error = err_m * 1000
                print(f"📐 Final: " + ", ".join([f"ch{ch}={deg}°" for ch, deg in sorted(servo_final.items())]))
                print(f"📏 Error: {error:.2f} mm")
                print("✅ Safe move complet")
                
            except ValueError:
                print("❌ Format: move <x_mm> <y_mm> <z_mm>  (ex: move 200 0 80)")
        
        
        elif cmd == "preset" and len(parts) >= 2:
            name = parts[1].lower()
            if name not in PRESETS:
                print(f"❌ Preset necunoscut. Disponibile: {', '.join(PRESETS.keys())}")
                continue
            
            preset = PRESETS[name]
            print(f"▶️  {preset['description']}")
            
            if preset.get("firmware_cmd"):
                controller.send_raw(preset["firmware_cmd"])
                current_ik_angles = np.zeros(len(chain.links))
            elif preset.get("target_xyz"):
                target = preset["target_xyz"]
                ik_angles, _ = solve_ik(chain, target, current_ik_angles)
                servo_angles = ik_to_servo_angles(ik_angles)
                controller.send_smooth(servo_angles, steps=move_steps, duration=move_duration)
                current_ik_angles = ik_angles
                print(f"📐 Channels: " + ", ".join([f"ch{ch}={deg}°" for ch, deg in sorted(servo_angles.items())]))
            
            print("✅ Done")
        
        elif cmd == "ch" and len(parts) == 3:
            try:
                ch = int(parts[1])
                deg = float(parts[2])
                if ch < 0 or ch > 6:
                    print("❌ Canal: 0-6")
                    continue
                if deg < 0 or deg > 180:
                    print("❌ Angle: 0-180°")
                    continue
                controller.send_angle(ch, deg)
                name = SERVO_CONFIG.get(ch, {}).get("name", f"Ch {ch}")
                print(f"✅ {name} (ch{ch}) → {deg}°")
            except ValueError:
                print("❌ Format: ch <canal> <grade>  (ex: ch 3 90)")
        
        elif cmd == "servo" and len(parts) == 3:
            # Backwards compatibility — redirect to ch
            print("ℹ️  Folosește 'ch' în loc de 'servo'. Redirectez...")
            try:
                ch = int(parts[1])
                deg = float(parts[2])
                controller.send_angle(ch, deg)
                name = SERVO_CONFIG.get(ch, {}).get("name", f"Ch {ch}")
                print(f"✅ {name} (ch{ch}) → {deg}°")
            except ValueError:
                print("❌ Format: ch <canal> <grade>")
        
        elif cmd == "show":
            plot_arm(chain, current_ik_angles, title="IRIS — Poziția curentă")
        
        elif cmd == "angles":
            servo_angles = ik_to_servo_angles(current_ik_angles)
            print("📐 Unghiuri curente:")
            for ch, deg in sorted(servo_angles.items()):
                cfg = SERVO_CONFIG[ch]
                print(f"   ch{ch} ({cfg['name']:12s}): {deg:6.1f}°")
        
        elif cmd == "reach":
            reach = (UPPER_ARM + FOREARM + WRIST + GRIPPER_LEN) * 1000
            print(f"📏 Arm reach estimates:")
            print(f"   Max reach (horizontal): ~{reach:.0f} mm")
            print(f"   Base height:            ~{BASE_HEIGHT*1000:.0f} mm")
            print(f"   Segments: {UPPER_ARM*1000:.0f} + {FOREARM*1000:.0f} + {WRIST*1000:.0f} + {GRIPPER_LEN*1000:.0f} mm")
            print(f"   Total link length:      ~{reach:.0f} mm")
        
        else:
            print("❌ Comandă necunoscută. Scrie 'help' sau una din: xyz, preset, servo, show, angles, reach, home, quit")


# ============================================================
# 7. DEMO AUTOMAT
# ============================================================

def run_demo(chain, controller):
    """Rulează un demo automat cu câteva poziții."""
    print("\n🤖 IRIS IK Demo — Secvență automată")
    print("="*50)
    
    # Home
    print("\n1️⃣  Home position...")
    home_angles = np.zeros(len(chain.links))
    servo = ik_to_servo_angles(home_angles)
    print(f"   Servo: {servo}")
    controller.send_all_angles(servo)
    time.sleep(1)
    
    # Test câteva poziții
    test_positions = [
        ("Forward reach",  [0.20, 0.00, 0.10]),
        ("Up high",        [0.05, 0.00, 0.28]),
        ("Low pick",       [0.18, 0.00, 0.03]),
        ("Left",           [0.00, 0.18, 0.10]),
        ("Right",          [0.00,-0.18, 0.10]),
        ("Center table",   [0.15, 0.00, 0.08]),
    ]
    
    prev_angles = home_angles
    
    for i, (name, target) in enumerate(test_positions, 2):
        print(f"\n{i}️⃣  {name}: target = ({target[0]*1000:.0f}, {target[1]*1000:.0f}, {target[2]*1000:.0f}) mm")
        
        ik_angles, ik_err = solve_ik(chain, target, prev_angles)
        
        # Forward kinematics check
        fk = chain.forward_kinematics(ik_angles)
        actual = fk[:3, 3]
        error = ik_err * 1000
        
        servo = ik_to_servo_angles(ik_angles)
        print(f"   Servo angles: {servo}")
        print(f"   Actual TCP:   ({actual[0]*1000:.1f}, {actual[1]*1000:.1f}, {actual[2]*1000:.1f}) mm")
        print(f"   Error:        {error:.2f} mm {'✅' if error < 5 else '⚠️'}")
        
        controller.send_all_angles(servo)
        prev_angles = ik_angles
        time.sleep(1.5)
    
    # Vizualizare ultima poziție
    print("\n📊 Afișez vizualizare 3D a ultimei poziții...")
    plot_arm(chain, prev_angles, target=test_positions[-1][1],
             title=f"IRIS — {test_positions[-1][0]}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="IRIS 6-DOF Arm — IK Demo")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (ex: /dev/ttyUSB0, /dev/cu.usbserial-*)")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Mod interactiv CLI")
    parser.add_argument("--no-plot", action="store_true",
                        help="Fără vizualizare matplotlib")
    args = parser.parse_args()
    
    # Build chain
    print("🔧 Building IRIS kinematic chain...")
    chain = build_iris_chain()
    print(f"   Links: {len(chain.links)} ({sum(chain.active_links_mask)} active)")
    print(f"   Segments: Base={BASE_HEIGHT*1000:.0f}mm, Upper={UPPER_ARM*1000:.0f}mm, "
          f"Forearm={FOREARM*1000:.0f}mm, Wrist={WRIST*1000:.0f}mm, Gripper={GRIPPER_LEN*1000:.0f}mm")
    
    # Serial connection
    if args.port:
        controller = SerialController(args.port, args.baud)
    else:
        print("ℹ️  No serial port — running in simulation mode")
        controller = DummyController()
    
    try:
        if args.interactive:
            interactive_mode(chain, controller)
        else:
            run_demo(chain, controller)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
