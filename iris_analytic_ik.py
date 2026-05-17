#!/usr/bin/env python3
"""
IRIS 6-DOF Robotic Arm — Analytic IK + Serial Control
======================================================
IK geometric simplu, fără biblioteci externe (doar math).
Calculează direct unghiurile servo din coordonate XYZ.

Canale PCA9685:
  ch0 = Gripper (SG90)           — open/close
  ch1 = Wrist Roll (MG90S)       — rotație stânga/dreapta
  ch2 = Wrist Pitch (MG90S)      — sus/jos mâna
  ch3 = Elbow (MG995)            — îndoaie cotul
  ch4 = Shoulder A (MG995)       — ridică/coboară brațul
  ch5 = Shoulder B (MG995)       — mirror automat în firmware
  ch6 = Base (MG995)             — rotație bază

Dimensiuni (mm, axă-la-axă):
  Base height:  75mm   (masă → ch4/5 shoulder axis)
  Upper arm:   152mm   (ch4/5 → ch3)
  Forearm:     136mm   (ch3 → ch2)
  Wrist:        30mm   (ch2 → ch1)
  Gripper:      38mm   (ch1 → TCP)

Abordare IK:
  - Forearm + wrist + gripper tratate ca un singur segment rigid (L2_total = 204mm)
  - ELBOW-UP configuration (alpha + beta)
  - ch2 (wrist pitch) folosit pentru fine-tune orientare gripper
  - FK verification integrat

Calibrare servo (28 apr 2026):
  ch1: offset=55°,  min=0°,  max=180°
  ch2: offset=170°, min=80°, max=180°
  ch3: offset=0°,   min=0°, max=155°  (0°=drept, crește=îndoaie)
  ch4: offset=150°, min=60°, max=150° (150°=orizontal, scade=ridică)
  ch6: offset=90°,  min=0°, max=180°
"""

import math
import time
import sys
import argparse

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ============================================================
# DIMENSIUNI (metri)
# ============================================================
BASE_H   = 0.105     # 105mm — masă → shoulder axis (măsurat)
L1       = 0.157     # upper arm: shoulder → elbow (157mm măsurat)
L2       = 0.136     # forearm: elbow → wrist pitch
L3       = 0.028     # wrist pitch → wrist roll (28mm măsurat)
L4       = 0.140     # wrist roll → TCP (140mm măsurat)
TOOL     = L3 + L4   # 68mm
L2_TOTAL = L2 + TOOL # 204mm — forearm+wrist+gripper ca segment rigid

# ============================================================
# SERVO CALIBRARE
# ============================================================
SERVO = {
    0: {"name": "Gripper",     "offset": 60,  "min": 0,   "max": 120},
    1: {"name": "Wrist Roll",  "offset": 55,  "min": 0,   "max": 180},
    2: {"name": "Wrist Pitch", "offset": 90,  "min": 0,   "max": 170},
    3: {"name": "Elbow",       "offset": 50,   "min": 20,   "max": 180},
    4: {"name": "Shoulder A",  "offset": 150, "min": 0,   "max": 150},
    5: {"name": "Shoulder B",  "offset": 30,  "min": 30,  "max": 120},
    6: {"name": "Base",        "offset": 90,  "min": 0,   "max": 180},
}

HOME = {0: 90, 1: 55, 2: 150, 3: 180, 4: 0, 5: 180, 6: 73}

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ============================================================
# IK ANALITIC — ELBOW-UP, L2_TOTAL
# ============================================================

def solve_ik(x_mm, y_mm, z_mm, wrist_angle=0):
    """
    Calculează unghiurile servo pentru a duce TCP-ul la (x, y, z) mm.
    
    Folosește 2-segment IK (L1, L2) pentru a poziționa wrist pivot,
    apoi TOOL (168mm) e orientat de ch2 (wrist pitch).
    
    wrist_angle: unghiul gripper-ului față de orizontal
        0° = gripper orizontal
       -90° = gripper vertical în jos (bun pentru pick-and-place)
       -60° = gripper înclinat în jos
    
    Returns: dict {channel: angle_degrees} sau None dacă nu poate ajunge
    """
    x = x_mm / 1000.0
    y = y_mm / 1000.0
    z = z_mm / 1000.0
    
    # === BAZA ===
    base_rad = math.atan2(y, x)
    base_servo = clamp(SERVO[6]["offset"] + math.degrees(base_rad),
                       SERVO[6]["min"], SERVO[6]["max"])
    
    # === PROIECȚIE ÎN PLAN VERTICAL ===
    r = math.sqrt(x*x + y*y)  # distanța orizontală
    
    # Wrist target angle
    wt = math.radians(wrist_angle)
    
    # Wrist pivot = TCP - TOOL * direction(wrist_angle)
    wx = r - TOOL * math.cos(wt)
    wz = (z - BASE_H) - TOOL * math.sin(wt)
    
    d = math.sqrt(wx*wx + wz*wz)
    
    # Verificare reach (cu L1 + L2, nu L2_TOTAL)
    if d > L1 + L2:
        print(f"  ⚠️  Wrist pivot prea departe (d={d*1000:.0f}mm > reach={(L1+L2)*1000:.0f}mm)")
        return None
    if d < abs(L1 - L2) + 0.001:
        print(f"  ⚠️  Wrist pivot prea aproape (d={d*1000:.0f}mm)")
        return None
    
    # === COSINE LAW pe L1, L2 ===
    cos_e = (L1*L1 + L2*L2 - d*d) / (2 * L1 * L2)
    cos_e = clamp(cos_e, -1, 1)
    theta_elbow = math.acos(cos_e)
    
    # === SHOULDER ANGLE ===
    gamma = math.pi - theta_elbow
    alpha = math.atan2(wz, wx)
    sin_beta = L2 * math.sin(gamma) / d
    beta = math.asin(clamp(sin_beta, -1, 1))
    
    # ELBOW-UP: shoulder = alpha + beta
    theta_shoulder = alpha + beta
    
    # === FOREARM ANGLE ===
    forearm_abs = theta_shoulder - (math.pi - theta_elbow)
    
    # === WRIST PITCH (ch2) ===
    # ch2 trebuie să roteasca TOOL de la forearm_abs la wrist_angle
    wrist_correction = wt - forearm_abs  # radiani
    
    # === CONVERSIE LA SERVO DEGREES ===
    
    # ch4 (Shoulder): offset=150 = orizontal, scade = ridică
    ch4_raw = SERVO[4]["offset"] - math.degrees(theta_shoulder)
    ch4 = clamp(round(ch4_raw, 1), SERVO[4]["min"], SERVO[4]["max"])
    
    # ch3 (Elbow): offset=50=drept, crește=îndoaie/coboară, scade=dreptește
    ch3_raw = SERVO[3]["offset"] + math.degrees(math.pi - theta_elbow)
    ch3 = clamp(round(ch3_raw, 1), SERVO[3]["min"], SERVO[3]["max"])
    
    # ch2 (Wrist Pitch): offset=170 = drept, scade = gripper urcă
    # wrist_correction > 0 = gripper mai sus decât forearm = ch2 scade
    ch2_raw = SERVO[2]["offset"] - math.degrees(wrist_correction)
    ch2 = clamp(round(ch2_raw, 1), SERVO[2]["min"], SERVO[2]["max"])
    
    # ch1 (Wrist Roll): rămâne la offset
    ch1 = SERVO[1]["offset"]
    
    # ch5 (Shoulder B): mirror
    ch5 = clamp(round(180 - ch4, 1), SERVO[5]["min"], SERVO[5]["max"])
    
    # Gripper angle (pentru info)
    gripper_angle = wrist_angle
    
    # Saturare check
    saturated = []
    clamped_critical = False
    if abs(ch4_raw - ch4) > 0.5:
        saturated.append(f"ch4(need {ch4_raw:.0f}°)")
        clamped_critical = True
    if abs(ch3_raw - ch3) > 0.5:
        saturated.append(f"ch3(need {ch3_raw:.0f}°)")
        clamped_critical = True
    if abs(ch2_raw - ch2) > 0.5:
        saturated.append(f"ch2(need {ch2_raw:.0f}°)")
        clamped_critical = True
    
    return {
        0: SERVO[0]["offset"], 1: ch1, 2: ch2, 3: ch3,
        4: ch4, 5: ch5, 6: round(base_servo, 1),
        "_gripper_angle": round(gripper_angle, 1),
        "_saturated": saturated,
        "_clamped": clamped_critical,
    }


def forward_k(angles):
    """
    Forward kinematics — calculează TCP din servo angles.
    Returns: (x_mm, y_mm, z_mm)
    """
    base_rad = math.radians(angles[6] - SERVO[6]["offset"])
    theta_s = math.radians(SERVO[4]["offset"] - angles[4])
    theta_e = math.pi - math.radians(angles[3] - SERVO[3]["offset"])
    
    # Elbow position
    ex = L1 * math.cos(theta_s)
    ez = L1 * math.sin(theta_s) + BASE_H
    
    # Forearm (L2 only, to wrist pivot)
    forearm = theta_s - (math.pi - theta_e)
    wpx = ex + L2 * math.cos(forearm)
    wpz = ez + L2 * math.sin(forearm)
    
    # Wrist correction → TCP angle (ch2 scade = gripper urcă = + angle)
    wrist_corr = math.radians(SERVO[2]["offset"] - angles[2])
    tcp_angle = forearm + wrist_corr
    
    # TCP (TOOL segment)
    tcpx = wpx + TOOL * math.cos(tcp_angle)
    tcpz = wpz + TOOL * math.sin(tcp_angle)
    
    # 3D
    tcp_X = tcpx * math.cos(base_rad) * 1000
    tcp_Y = tcpx * math.sin(base_rad) * 1000
    tcp_Z = tcpz * 1000
    
    return round(tcp_X, 1), round(tcp_Y, 1), round(tcp_Z, 1)


# ============================================================
# SERIAL CONTROLLER
# ============================================================

class SerialController:
    def __init__(self, port, baud=115200):
        if not HAS_SERIAL:
            raise RuntimeError("pyserial not installed: pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)
        self.ser.reset_input_buffer()
        print(f"✅ Conectat la {port}")
    
    def send(self, channel, angle, smooth=True):
        """Trimite un servo direct la unghi."""
        cmd = f"ch {channel} {int(angle)}\n"
        self.ser.write(cmd.encode())
        time.sleep(0.02)
    
    def send_smooth(self, target_angles, steps=30, duration=1.0):
        """
        Mișcare smooth — interpolează liniar de la pozițiile curente 
        la target în N pași pe durata specificată.
        """
        # Canalele valide
        channels = sorted(k for k in target_angles.keys() if isinstance(k, int) and k != 5)
        
        # Poziții curente (din current_angles global)
        global current_angles
        start = {}
        for ch in channels:
            start[ch] = current_angles.get(ch, target_angles[ch])
        
        step_delay = duration / steps
        
        for i in range(1, steps + 1):
            t = i / steps  # 0.0 → 1.0
            # Ease in-out (smooth)
            t = t * t * (3 - 2 * t)
            
            for ch in channels:
                val = start[ch] + (target_angles[ch] - start[ch]) * t
                cmd = f"ch {ch} {int(val)}\n"
                self.ser.write(cmd.encode())
            time.sleep(step_delay)
    
    def send_all(self, angles_dict, smooth=True):
        """Trimite toate servo-urile — smooth sau direct."""
        if smooth:
            self.send_smooth(angles_dict, steps=30, duration=current_speed)
        else:
            for ch in sorted(k for k in angles_dict.keys() if isinstance(k, int)):
                if ch == 5:
                    continue
                self.send(ch, angles_dict[ch])
    
    def close(self):
        self.ser.close()
        print("🔌 Serial închis")


class DummyController:
    def send(self, channel, angle, smooth=True):
        pass
    def send_smooth(self, target_angles, steps=30, duration=1.0):
        pass
    def send_all(self, angles_dict, smooth=True):
        pass
    def close(self):
        pass


# ============================================================
# STATE
# ============================================================
current_angles = dict(HOME)
current_speed = 2.0
wrist_angle = -10  # gripper aproape orizontal, ușor în jos
presets = {"home": dict(HOME)}  # preset-uri salvate

PRESETS_FILE = "iris_presets.json"

def load_presets():
    global presets
    import json, os
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE) as f:
                presets = json.load(f)
                # Convert string keys back to int
                for name in presets:
                    presets[name] = {int(k): v for k, v in presets[name].items()}
            print(f"📂 {len(presets)} preset-uri încărcate")
        except:
            pass
    presets["home"] = dict(HOME)

def save_presets():
    import json
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)


# ============================================================
# INTERACTIVE CLI
# ============================================================

def interactive(ctrl):
    global current_angles, current_speed, wrist_angle, presets
    
    load_presets()
    
    print("\n🤖 IRIS Analytic IK — Interactive Control")
    print("=" * 50)
    print("Comenzi:")
    print("  xyz X Y Z       — mergi la coordonate (mm)")
    print("  ch N ANGLE       — setează servo N la ANGLE°")
    print("  home             — poziție home")
    print("  start            — tranziție sigură home → IK")
    print("  grip / open      — închide / deschide gripper")
    print("  speed S          — durata mișcării în secunde (default 2)")
    print("  wrist W          — ajustare wrist pitch")
    print("  show             — arată starea curentă")
    print("  fk               — forward kinematics")
    print("  test             — test sequence")
    print("  pick X Y Z       — secvență pick")
    print("  place X Y Z      — secvență place")
    print("  save NUME        — salvează poziția curentă ca preset")
    print("  go NUME          — du-te la preset")
    print("  presets           — arată toate preset-urile")
    print("  del NUME         — șterge un preset")
    print("  quit             — ieșire")
    print()
    
    while True:
        try:
            raw = input("IRIS> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not raw:
            continue
        
        parts = raw.split()
        cmd = parts[0].lower()
        
        if cmd in ("quit", "q"):
            break
        
        elif cmd == "home":
            print("🏠 Home position")
            ctrl.send_all(dict(HOME))
            current_angles = dict(HOME)
        
        elif cmd == "start":
            print("🚀 Tranziție sigură → IK ready")
            steps = [
                {0: 60, 1: 55, 2: 130, 3: 60, 4: 80, 5: 100, 6: 90},  # shoulder sus, elbow îndoit
                {0: 60, 1: 55, 2: 110, 3: 80, 4: 75, 5: 105, 6: 90},  # pregătire IK
            ]
            for s in steps:
                ctrl.send_all(s)
                time.sleep(1.0)
            current_angles = steps[-1]
            print("✅ Ready pentru xyz")
        
        elif cmd == "xyz" and len(parts) == 4:
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                print("❌ Format: xyz X Y Z (mm)")
                continue
            
            # Compensare empiric (calibrat din măsurători)
            x_ik = x - 60
            z_ik = (z + 370) / 2.07
            
            result = solve_ik(x_ik, y, z_ik, wrist_angle)
            if result is None:
                print("❌ Nu poate ajunge acolo")
                continue
            
            fk = forward_k(result)
            err = math.sqrt((fk[0]-x)**2 + (fk[1]-y)**2 + (fk[2]-z)**2)
            
            print(f"🎯 Target: ({x:.0f}, {y:.0f}, {z:.0f}) mm")
            print(f"📐 ch4={result[4]}° ch3={result[3]}° ch2={result[2]}° ch6={result[6]}°")
            print(f"📍 FK: ({fk[0]}, {fk[1]}, {fk[2]}) mm  |  err={err:.1f}mm")
            print(f"🤚 Gripper angle: {result['_gripper_angle']}° de la orizontal")
            
            if result["_saturated"]:
                print(f"⚠️  Servo saturate: {', '.join(result['_saturated'])}")
            if result.get("_clamped"):
                print(f"🚨 ATENȚIE: servo clampat → poziția reală NU va fi ({x:.0f},{y:.0f},{z:.0f})!")
                print(f"   FK spune TCP va fi la ({fk[0]:.0f},{fk[1]:.0f},{fk[2]:.0f}) — err={err:.0f}mm")
            
            ctrl.send_all(result)
            current_angles = result
            print("✅ Trimis la ESP32")
        
        elif cmd == "ch" and len(parts) == 3:
            try:
                ch = int(parts[1])
                ang = float(parts[2])
            except ValueError:
                print("❌ Format: ch CHANNEL ANGLE")
                continue
            if ch not in SERVO:
                print(f"❌ Canal invalid. Valid: {list(SERVO.keys())}")
                continue
            ang = clamp(ang, SERVO[ch]["min"], SERVO[ch]["max"])
            target = dict(current_angles)
            target[ch] = ang
            ctrl.send_all(target)
            current_angles[ch] = ang
            print(f"✅ ch{ch} ({SERVO[ch]['name']}) → {ang}°")
        
        elif cmd == "grip":
            ctrl.send(0, 120)
            current_angles[0] = 120
            print("✊ Gripper CLOSED")
        
        elif cmd == "open":
            ctrl.send(0, 30)
            current_angles[0] = 30
            print("🤚 Gripper OPEN")
        
        elif cmd == "speed" and len(parts) == 2:
            try:
                current_speed = float(parts[1])
                print(f"🏎️  Speed: {current_speed}")
            except ValueError:
                print("❌ Format: speed N")
        
        elif cmd == "wrist" and len(parts) == 2:
            try:
                wrist_angle = float(parts[1])
                print(f"🤚 Wrist adjust: {wrist_angle}° (adaugat la ch2 offset)")
            except ValueError:
                print("❌ Format: wrist ANGLE")
        
        elif cmd == "show":
            print(f"📐 Angles: ch0={current_angles.get(0)}, ch1={current_angles.get(1)}, "
                  f"ch2={current_angles.get(2)}, ch3={current_angles.get(3)}, "
                  f"ch4={current_angles.get(4)}, ch6={current_angles.get(6)}")
            print(f"🏎️  Speed: {current_speed}")
            print(f"🤚 Wrist adjust: {wrist_angle}°")
        
        elif cmd == "fk":
            fk = forward_k(current_angles)
            print(f"📍 TCP: ({fk[0]}, {fk[1]}, {fk[2]}) mm")
        
        elif cmd == "test":
            print("🧪 Test sequence...")
            test_points = [
                (200, 0, 100, "forward low"),
                (200, 0, 150, "forward mid"),
                (150, 0, 200, "close high"),
                (150, 0, 100, "close low"),
                (200, 0, 80,  "forward very low"),
                (180, 0, 50,  "near table"),
                (100, 100, 150, "diagonal"),
            ]
            print(f"  {'Target':<16} {'Desc':15} {'ch4':>5} {'ch3':>5} {'ch2':>5} {'FK Z':>6} {'err':>5} {'grip°':>6}")
            print("  " + "-" * 80)
            for tx, ty, tz, desc in test_points:
                r = solve_ik(tx, ty, tz, wrist_angle)
                if r:
                    fk = forward_k(r)
                    err = math.sqrt((fk[0]-tx)**2 + (fk[1]-ty)**2 + (fk[2]-tz)**2)
                    ok = "✅" if err < 3 else ("⚠️" if err < 10 else "❌")
                    print(f"  ({tx:>3},{ty:>3},{tz:>3}) {desc:15} {r[4]:>5.1f} {r[3]:>5.1f} {r[2]:>5.1f} {fk[2]:>6.1f} {err:>4.1f}  {r['_gripper_angle']:>+5.1f}° {ok}")
                else:
                    print(f"  ({tx:>3},{ty:>3},{tz:>3}) {desc:15} UNREACHABLE ❌")
        
        elif cmd == "pick" and len(parts) == 4:
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                print("❌ Format: pick X Y Z")
                continue
            safe_z = max(z + 80, 180)
            print(f"🏗️  Pick sequence: approach at Z={safe_z}mm → descend to Z={z}mm → grip")
            # Open gripper
            ctrl.send(0, 30)
            time.sleep(0.3)
            # Go above
            r = solve_ik(x, y, safe_z, wrist_angle)
            if r:
                ctrl.send_all(r)
                time.sleep(1.0)
            # Descend
            r = solve_ik(x, y, z, wrist_angle)
            if r:
                ctrl.send_all(r)
                current_angles = r
                time.sleep(0.8)
            # Grip
            ctrl.send(0, 120)
            time.sleep(0.5)
            # Lift
            r = solve_ik(x, y, safe_z, wrist_angle)
            if r:
                ctrl.send_all(r)
                time.sleep(0.5)
            print("✅ Pick done")
        
        elif cmd == "place" and len(parts) == 4:
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                print("❌ Format: place X Y Z")
                continue
            safe_z = max(z + 80, 180)
            print(f"📦 Place sequence: approach at Z={safe_z}mm → descend to Z={z}mm → release")
            # Go above
            r = solve_ik(x, y, safe_z, wrist_angle)
            if r:
                ctrl.send_all(r)
                time.sleep(1.0)
            # Descend
            r = solve_ik(x, y, z, wrist_angle)
            if r:
                ctrl.send_all(r)
                current_angles = r
                time.sleep(0.8)
            # Release
            ctrl.send(0, 30)
            time.sleep(0.5)
            # Lift
            r = solve_ik(x, y, safe_z, wrist_angle)
            if r:
                ctrl.send_all(r)
                time.sleep(0.5)
            print("✅ Place done")
        
        elif cmd == "save" and len(parts) >= 2:
            name = parts[1].lower()
            # Salvează doar canalele servo (nu cheile cu _)
            angles_to_save = {k: v for k, v in current_angles.items() if isinstance(k, int)}
            presets[name] = angles_to_save
            save_presets()
            print(f"💾 Preset '{name}' salvat: {angles_to_save}")
        
        elif cmd == "go" and len(parts) >= 2:
            name = parts[1].lower()
            if name in presets:
                print(f"🎯 Go to preset '{name}'")
                ctrl.send_all(presets[name])
                current_angles = dict(presets[name])
                print(f"✅ Done")
            else:
                print(f"❌ Preset '{name}' nu există. Scrie 'presets' să vezi lista.")
        
        elif cmd == "presets":
            if presets:
                print(f"📋 Preset-uri salvate ({len(presets)}):")
                for name, angles in sorted(presets.items()):
                    vals = " ".join(f"ch{k}={v}°" for k, v in sorted(angles.items()))
                    print(f"  {name:15s} → {vals}")
            else:
                print("📋 Niciun preset salvat.")
        
        elif cmd == "del" and len(parts) >= 2:
            name = parts[1].lower()
            if name == "home":
                print("❌ Nu poți șterge preset-ul 'home'")
            elif name in presets:
                del presets[name]
                save_presets()
                print(f"🗑️  Preset '{name}' șters")
            else:
                print(f"❌ Preset '{name}' nu există")
        
        elif cmd == "help":
            print("Comenzi: xyz, ch, home, start, grip, open, speed, wrist, show, fk, test, pick, place, save, go, presets, del, quit")
        
        else:
            print(f"❌ Comandă necunoscută: '{cmd}'. Scrie 'help'.")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="IRIS Analytic IK")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (ex: /dev/cu.usbserial-0001)")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()
    
    if args.port:
        ctrl = SerialController(args.port, args.baud)
    else:
        print("ℹ️  Mod simulare (fără serial)")
        ctrl = DummyController()
    
    try:
        interactive(ctrl)
    finally:
        ctrl.close()


if __name__ == "__main__":
    main()
