#!/usr/bin/env python3
"""
IRIS Agent v1.0 — Level 3: Gemini Robotics-ER 1.6 as the Brain
═══════════════════════════════════════════════════════════════

Gemini Robotics-ER 1.6 orchestrates everything:
  • Sees through the camera (pointing, object detection)
  • Plans multi-step tasks (pick & place, sorting, etc.)
  • Calls functions: move arm, gripper, detect objects, check success
  • Text chat from terminal

Architecture:
  User (text) → Gemini ER 1.6 (brain) → Function Calls → Bridge → ESP32

Requirements:
  pip install requests --break-system-packages

Usage:
  export GEMINI_API_KEY=your_key
  python iris_bridge.py &
  python iris_agent.py
"""

import os
import sys
import json
import time
import base64
import threading
import requests
import urllib.request
from datetime import datetime

# ─── CONFIG ───
BRIDGE_URL = "http://localhost:8765"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-robotics-er-1.6-preview"  # single model for everything
CAMERA_SNAPSHOT_URL = f"{BRIDGE_URL}/camera/snapshot"
GRIPPER_URL = f"{BRIDGE_URL}/gripper"
IK_URL = f"{BRIDGE_URL}/ik"
SERVO_URL = f"{BRIDGE_URL}/servo"
SERVOS_URL = f"{BRIDGE_URL}/servos"
PING_URL = f"{BRIDGE_URL}/ping"

# Arm defaults
SAFE_Z = 200       # mm — safe height for transit
PICK_Z = 50        # mm — default pick height hint (Gemini can go lower, min 10mm)
PLACE_Z = 80       # mm — default place height
HOME_POS = {"x": 200, "y": 0, "z": 250}

# ─── COLORS for terminal ───
class C:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# ══════════════════════════════════════════
#  ROBOT FUNCTIONS (called by Gemini)
# ══════════════════════════════════════════

def move_arm(x: float, y: float, z: float) -> dict:
    """Move the robot arm to XYZ position in millimeters using inverse kinematics.
    
    Args:
        x: Forward distance from base in mm (typically 80-350)
        y: Left/right offset in mm (negative=right, positive=left, typically -120 to 120)  
        z: Height in mm (0=table level, 200=safe transit height)
    
    Returns:
        Dict with success status, actual position achieved, and IK error in mm.
    """
    try:
        r = requests.post(IK_URL, json={"x": x, "y": y, "z": z, "send": True}, timeout=5)
        d = r.json()
        if d.get("ok"):
            actual = d.get("actual", {})
            print(f"  {C.GREEN}→ Arm moved to ({actual.get('x', x):.0f}, {actual.get('y', y):.0f}, {actual.get('z', z):.0f})mm  err={d.get('error_mm', 0):.1f}mm{C.RESET}")
            return {"success": True, "actual": actual, "error_mm": d.get("error_mm", 0)}
        else:
            print(f"  {C.RED}✗ IK failed: {d.get('error', 'unknown')}{C.RESET}")
            return {"success": False, "error": d.get("error", "IK solve failed")}
    except Exception as e:
        print(f"  {C.RED}✗ Move failed: {e}{C.RESET}")
        return {"success": False, "error": str(e)}


def open_gripper() -> dict:
    """Open the gripper to release an object. Moves instantly at maximum speed.
    
    Returns:
        Dict with success status.
    """
    try:
        r = requests.get(f"{GRIPPER_URL}?angle=0", timeout=3)
        d = r.json()
        print(f"  {C.GREEN}→ Gripper OPEN (0°){C.RESET}")
        return {"success": True, "state": "open", "angle": 0}
    except Exception as e:
        print(f"  {C.RED}✗ Gripper failed: {e}{C.RESET}")
        return {"success": False, "error": str(e)}


def close_gripper() -> dict:
    """Close the gripper to grab an object. Moves instantly at maximum speed.
    
    Returns:
        Dict with success status.
    """
    try:
        r = requests.get(f"{GRIPPER_URL}?angle=70", timeout=3)
        d = r.json()
        print(f"  {C.GREEN}→ Gripper CLOSED (70°){C.RESET}")
        return {"success": True, "state": "closed", "angle": 70}
    except Exception as e:
        print(f"  {C.RED}✗ Gripper failed: {e}{C.RESET}")
        return {"success": False, "error": str(e)}


def move_to_safe_height() -> dict:
    """Move the arm up to a safe transit height to avoid collisions.
    Keeps current X/Y and raises Z to safe level.
    
    Returns:
        Dict with success status.
    """
    return move_arm(HOME_POS["x"], HOME_POS["y"], SAFE_Z)


def go_home() -> dict:
    """Move the arm to its home/rest position (centered, raised).
    
    Returns:
        Dict with success status.
    """
    open_gripper()
    time.sleep(0.3)
    return move_arm(HOME_POS["x"], HOME_POS["y"], HOME_POS["z"])


def detect_objects(query: str) -> dict:
    """Use the camera to detect objects on the table using Gemini Robotics-ER vision.
    
    Takes a camera snapshot, sends it to Gemini ER for pointing/detection,
    and returns object positions in robot coordinates (mm).
    
    Args:
        query: What to look for, e.g. "red cube", "all objects", "the pen closest to the cup"
    
    Returns:
        Dict with list of detected objects, each having label, pixel coords, and robot XYZ coords.
    """
    try:
        # Get camera snapshot
        r = requests.get(CAMERA_SNAPSHOT_URL, timeout=5)
        if r.status_code != 200:
            return {"success": False, "error": "Camera not available. Start camera first."}
        
        image_bytes = r.content
        img_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Parse real image dimensions from JPEG
        img_w, img_h = 640, 480  # fallback
        try:
            # JPEG SOF0 marker parsing
            data = image_bytes
            i = 2
            while i < len(data) - 1:
                if data[i] == 0xFF:
                    marker = data[i+1]
                    if marker in (0xC0, 0xC1, 0xC2):  # SOF markers
                        img_h = (data[i+5] << 8) | data[i+6]
                        img_w = (data[i+7] << 8) | data[i+8]
                        break
                    elif marker == 0xD9:  # EOI
                        break
                    elif marker in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0x01):
                        i += 2
                    else:
                        seg_len = (data[i+2] << 8) | data[i+3]
                        i += 2 + seg_len
                else:
                    i += 1
        except:
            pass
        print(f"  {C.DIM}  Camera: {img_w}x{img_h}{C.RESET}")
        
        # Query Gemini ER for pointing
        prompt = f"""{query}
Point to each relevant object. The answer should follow the json format: [{{"point": [y, x], "label": <label>}}].
The points are in [y, x] format normalized to 0-1000."""
        
        request_body = {
            "contents": [{"parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                {"text": prompt}
            ]}],
            "generationConfig": {
                "temperature": 0.5,
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url, data=json.dumps(request_body).encode('utf-8'),
                                     headers={"Content-Type": "application/json"}, method="POST")
        
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        
        # Extract text
        text = ""
        for candidate in result.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    text += part["text"]
        
        # Parse JSON points
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()
        
        try:
            raw_points = json.loads(clean)
        except json.JSONDecodeError:
            print(f"  {C.YELLOW}⚠ Gemini returned non-JSON: {text[:100]}{C.RESET}")
            return {"success": True, "objects": [], "raw_response": text}
        
        # Convert normalized points to robot coords via bridge
        objects = []
        for p in raw_points:
            if "point" not in p:
                continue
            ny, nx = p["point"]
            px = int(nx * img_w / 1000)
            py = int(ny * img_h / 1000)
            
            # Use bridge's vision/click to convert pixel → robot coords
            try:
                vr = requests.post(f"{BRIDGE_URL}/vision/click",
                                   json={"px": px, "py": py, "z": PICK_Z, "send": False}, timeout=3)
                vd = vr.json()
                if vd.get("ok"):
                    robot = vd.get("robot", {})
                    objects.append({
                        "label": p.get("label", "unknown"),
                        "pixel": {"x": px, "y": py},
                        "robot": robot,
                        "norm": p["point"]
                    })
                    print(f"  {C.CYAN}  • {p.get('label', '?')}: px({px},{py}) → robot({robot.get('x', 0):.0f}, {robot.get('y', 0):.0f})mm{C.RESET}")
            except:
                pass
        
        print(f"  {C.GREEN}→ Detected {len(objects)} objects{C.RESET}")
        return {"success": True, "objects": objects, "count": len(objects)}
        
    except Exception as e:
        print(f"  {C.RED}✗ Detection failed: {e}{C.RESET}")
        return {"success": False, "error": str(e)}


def check_success(task_description: str) -> dict:
    """Take a camera snapshot and check if a task was completed successfully.
    
    Uses Gemini ER to visually verify if the described task appears done.
    
    Args:
        task_description: What should have happened, e.g. "the red cube is now in the bowl"
    
    Returns:
        Dict with success assessment, confidence, and reasoning.
    """
    try:
        r = requests.get(CAMERA_SNAPSHOT_URL, timeout=5)
        if r.status_code != 200:
            return {"success": False, "error": "Camera not available"}
        
        img_b64 = base64.b64encode(r.content).decode('utf-8')
        
        prompt = f"""Look at this image from a robot's camera. 
Determine if the following task has been completed successfully: "{task_description}"

Respond in JSON format:
{{"completed": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""
        
        request_body = {
            "contents": [{"parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                {"text": prompt}
            ]}],
            "generationConfig": {
                "temperature": 0.3,
                "thinkingConfig": {"thinkingBudget": 1024}
            }
        }
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url, data=json.dumps(request_body).encode('utf-8'),
                                     headers={"Content-Type": "application/json"}, method="POST")
        
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        
        text = ""
        for candidate in result.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    text += part["text"]
        
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            clean = clean.rsplit("```", 1)[0]
        
        try:
            check = json.loads(clean.strip())
            completed = check.get("completed", False)
            confidence = check.get("confidence", 0)
            reasoning = check.get("reasoning", "")
            status = "✓" if completed else "✗"
            color = C.GREEN if completed else C.RED
            print(f"  {color}{status} Success check: {reasoning} (confidence: {confidence:.0%}){C.RESET}")
            return {"success": True, **check}
        except:
            return {"success": True, "completed": False, "confidence": 0, "reasoning": text[:200]}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def wait_seconds(seconds: float) -> dict:
    """Wait for a specified number of seconds. Useful between movements to let servos settle.
    
    Args:
        seconds: Number of seconds to wait (0.1 to 10)
    
    Returns:
        Dict confirming the wait.
    """
    seconds = max(0.1, min(10, seconds))
    time.sleep(seconds)
    print(f"  {C.DIM}  ⏳ Waited {seconds}s{C.RESET}")
    return {"success": True, "waited": seconds}


def describe_scene() -> dict:
    """Take a camera snapshot and get a general description of what's on the table.
    
    Returns:
        Dict with scene description and list of visible objects.
    """
    try:
        r = requests.get(CAMERA_SNAPSHOT_URL, timeout=5)
        if r.status_code != 200:
            return {"success": False, "error": "Camera not available"}
        
        img_b64 = base64.b64encode(r.content).decode('utf-8')
        
        prompt = """Describe what you see on this table/workspace from a robot arm's camera.
List all visible objects with approximate positions (left, center, right, near, far).
Be concise but thorough. Respond in the user's language if they spoke Romanian, otherwise English."""
        
        request_body = {
            "contents": [{"parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                {"text": prompt}
            ]}],
            "generationConfig": {
                "temperature": 0.5,
                "thinkingConfig": {"thinkingBudget": 512}
            }
        }
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url, data=json.dumps(request_body).encode('utf-8'),
                                     headers={"Content-Type": "application/json"}, method="POST")
        
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        
        text = ""
        for candidate in result.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    text += part["text"]
        
        print(f"  {C.CYAN}→ Scene: {text[:150]}...{C.RESET}")
        return {"success": True, "description": text}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════
#  TOOL DEFINITIONS for Gemini
# ══════════════════════════════════════════

TOOLS = {
    "move_arm": move_arm,
    "open_gripper": open_gripper,
    "close_gripper": close_gripper,
    "move_to_safe_height": move_to_safe_height,
    "go_home": go_home,
    "detect_objects": detect_objects,
    "check_success": check_success,
    "wait_seconds": wait_seconds,
    "describe_scene": describe_scene,
}

TOOL_DECLARATIONS = [
    {
        "name": "move_arm",
        "description": "Move the robot arm to an XYZ position in millimeters using inverse kinematics. X is forward (80-350mm), Y is left/right (-120 to 120mm, negative=right), Z is height (0=table, 200=safe).",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Forward distance from base in mm (80-350)"},
                "y": {"type": "number", "description": "Left/right offset in mm (-120 to 120, negative=right)"},
                "z": {"type": "number", "description": "Height in mm (0=table surface, 30=small object grab, 200=safe transit). Go as low as needed!"}
            },
            "required": ["x", "y", "z"]
        }
    },
    {
        "name": "open_gripper",
        "description": "Open the gripper to release an object. Instant movement at max speed.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "close_gripper", 
        "description": "Close the gripper to grab an object. Instant movement at max speed.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "move_to_safe_height",
        "description": "Move arm to safe transit height (Z=200mm) to avoid collisions during lateral movement.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "go_home",
        "description": "Move arm to home/rest position. Opens gripper and centers arm raised.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "detect_objects",
        "description": "Use camera + Gemini ER vision to find objects on the table. Returns each object's label and XYZ robot coordinates in mm. Use this BEFORE moving to an object.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to detect, e.g. 'red cube', 'all objects on table', 'the pen closest to the cup'"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "check_success",
        "description": "Take a photo and verify if a task was completed. Use AFTER performing an action to confirm success.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_description": {"type": "string", "description": "What should have happened, e.g. 'the red cube is in the bowl'"}
            },
            "required": ["task_description"]
        }
    },
    {
        "name": "wait_seconds",
        "description": "Wait for servos to finish moving. Use between arm movements (0.5-2s typical).",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "Seconds to wait (0.1-10)"}
            },
            "required": ["seconds"]
        }
    },
    {
        "name": "describe_scene",
        "description": "Take a photo and describe everything visible on the table. Good for initial scene understanding.",
        "parameters": {"type": "object", "properties": {}}
    },
]


# ══════════════════════════════════════════
#  GEMINI CHAT AGENT (text mode)
# ══════════════════════════════════════════

SYSTEM_PROMPT = """You are IRIS — an intelligent robot arm assistant. You control a 6-DOF robot arm with a gripper.

CAPABILITIES:
- You can see through a camera mounted above the workspace
- You can detect and locate objects using vision (detect_objects)
- You can move your arm to any XYZ coordinate using inverse kinematics (move_arm)
- You can grab objects (close_gripper) and release them (open_gripper)
- You can verify if tasks succeeded (check_success)
- You can describe what you see (describe_scene)

WORKSPACE:
- X axis: 80-350mm (forward from base)
- Y axis: -120 to 120mm (right to left)  
- Z axis: 0mm (table surface) to 250mm. You can go as low as Z=0mm.
- 200mm is a good safe transit height
- Choose Z based on object height: flat objects need Z=0-10, small items Z=10-30, tall items Z=30-60
- Always move to safe Z (200mm) before lateral moves to avoid hitting objects

PICK & PLACE PROCEDURE:
1. detect_objects to find the target
2. move_arm to ABOVE the object (same X,Y but Z=200)
3. wait_seconds(3) — IMPORTANT: servos are slow, big Z moves need 3 seconds!
4. open_gripper
5. move_arm DOWN to the object — choose Z based on object size! Go LOW enough to grab it (Z=0-30 for most objects)
6. wait_seconds(3) — MUST wait for arm to physically arrive before gripping!
7. close_gripper
8. wait_seconds(1) — let gripper fully close
9. move_arm UP (Z=200) — lift with object
10. wait_seconds(3)
11. move_arm to ABOVE destination (Z=200)
12. wait_seconds(2)
13. move_arm DOWN to place height
14. wait_seconds(3)
15. open_gripper — release
16. wait_seconds(1)
17. move_arm UP (Z=200) — clear
18. check_success to verify

CRITICAL: The servos move slowly (smooth interpolation). A move from Z=200 to Z=10 takes ~3 seconds.
If you close the gripper before the arm arrives, you will grab AIR. Always wait_seconds(3) after big moves!

PERSONALITY:
- You speak Romanian naturally (the user is Romanian) but understand English too
- You're enthusiastic about helping but precise about movements
- If unsure about an object's position, always detect_objects first
- If a pick fails, retry once with slightly adjusted coordinates
- Always confirm what you're about to do before executing complex tasks
- NEVER rush — always wait enough for the arm to physically reach its target
"""


def gemini_chat(user_message: str, conversation_history: list) -> str:
    """Send a message to Gemini with function calling enabled.
    Handles the full tool-use loop: call → execute → return result → get response."""
    
    if not GEMINI_API_KEY:
        return "❌ GEMINI_API_KEY not set! Run: export GEMINI_API_KEY=your_key"
    
    # Add user message to history
    conversation_history.append({"role": "user", "parts": [{"text": user_message}]})
    
    # API endpoint — Gemini Robotics-ER 1.6 for everything
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    max_turns = 15  # max function call rounds
    
    for turn in range(max_turns):
        request_body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": conversation_history,
            "tools": [{"functionDeclarations": TOOL_DECLARATIONS}],
            "generationConfig": {
                "temperature": 0.7,
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(request_body).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            return f"❌ Gemini API error: {e}"
        
        # Parse response
        candidates = result.get("candidates", [])
        if not candidates:
            return "❌ No response from Gemini"
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        
        # Check for function calls
        function_calls = [p for p in parts if "functionCall" in p]
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        
        # Add model response to history
        conversation_history.append(content)
        
        if not function_calls:
            # No more function calls — return text response
            return " ".join(text_parts).strip()
        
        # Execute function calls
        function_responses = []
        for fc_part in function_calls:
            fc = fc_part["functionCall"]
            fn_name = fc["name"]
            fn_args = fc.get("args", {})
            
            print(f"\n  {C.MAGENTA}⚡ {fn_name}({json.dumps(fn_args, ensure_ascii=False)}){C.RESET}")
            
            if fn_name in TOOLS:
                try:
                    fn_result = TOOLS[fn_name](**fn_args)
                except Exception as e:
                    fn_result = {"error": str(e)}
            else:
                fn_result = {"error": f"Unknown function: {fn_name}"}
            
            function_responses.append({
                "functionResponse": {
                    "name": fn_name,
                    "response": fn_result
                }
            })
        
        # Add function responses to history  
        conversation_history.append({"role": "user", "parts": function_responses})
    
    return "⚠ Max function call rounds reached"


# ══════════════════════════════════════════
#  MAIN — Text Chat Mode
# ══════════════════════════════════════════

def check_bridge():
    """Check if bridge is running and what's available."""
    try:
        r = requests.get(PING_URL, timeout=2)
        d = r.json()
        serial_ok = d.get("serial", False)
        vision_ok = d.get("vision", False)
        return True, serial_ok, vision_ok
    except:
        return False, False, False


def main():
    print()
    print(f"{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}║  IRIS Agent v1.0 — Gemini ER 1.6 Brain       ║{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}║  Chat → Gemini ER → IK → ESP32               ║{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}╚══════════════════════════════════════════════╝{C.RESET}")
    print()
    
    # Check API key
    if not GEMINI_API_KEY:
        print(f"{C.RED}✗ GEMINI_API_KEY not set!{C.RESET}")
        print(f"  Run: export GEMINI_API_KEY=your_key")
        sys.exit(1)
    print(f"{C.GREEN}✓{C.RESET} Gemini API key loaded")
    print(f"  Model: {GEMINI_MODEL}")
    
    # Check bridge
    bridge_ok, serial_ok, vision_ok = check_bridge()
    if bridge_ok:
        print(f"{C.GREEN}✓{C.RESET} Bridge connected at {BRIDGE_URL}")
        print(f"  Serial: {'✓' if serial_ok else '✗ (test mode)'}")
        print(f"  Vision: {'✓ calibrated' if vision_ok else '✗ not calibrated'}")
    else:
        print(f"{C.RED}✗{C.RESET} Bridge not running! Start iris_bridge.py first.")
        print(f"  Continuing in test mode (function calls will fail)")
    
    # Start camera if bridge is up
    if bridge_ok:
        try:
            requests.get(f"{BRIDGE_URL}/camera/start", timeout=3)
            print(f"{C.GREEN}✓{C.RESET} Camera started")
        except:
            print(f"{C.YELLOW}⚠{C.RESET} Camera failed to start")
    
    print()
    print(f"{C.BOLD}Available tools:{C.RESET}")
    for td in TOOL_DECLARATIONS:
        print(f"  {C.MAGENTA}⚡{C.RESET} {td['name']} — {td['description'][:60]}...")
    
    print()
    print(f"{C.BOLD}Commands:{C.RESET}")
    print(f"  Type a message to talk to IRIS")
    print(f"  {C.DIM}/clear{C.RESET}  — reset conversation")
    print(f"  {C.DIM}/home{C.RESET}   — send arm home")
    print(f"  {C.DIM}/look{C.RESET}   — describe scene")
    print(f"  {C.DIM}/quit{C.RESET}   — exit")
    print()
    
    conversation_history = []
    
    while True:
        try:
            user_input = input(f"{C.BOLD}{C.GREEN}Tu ❯ {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}La revedere!{C.RESET}")
            break
        
        if not user_input:
            continue
        
        # Quick commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print(f"{C.DIM}La revedere!{C.RESET}")
            break
        
        if user_input.lower() == "/clear":
            conversation_history = []
            print(f"{C.YELLOW}✓ Conversație resetată{C.RESET}")
            continue
        
        if user_input.lower() == "/home":
            go_home()
            continue
        
        if user_input.lower() == "/look":
            result = describe_scene()
            if result.get("success"):
                print(f"\n{C.CYAN}IRIS:{C.RESET} {result['description']}\n")
            continue
        
        # Send to Gemini
        print(f"\n{C.DIM}  Gemini gândește...{C.RESET}")
        start = time.time()
        
        response = gemini_chat(user_input, conversation_history)
        
        elapsed = time.time() - start
        print(f"\n{C.CYAN}{C.BOLD}IRIS:{C.RESET} {response}")
        print(f"{C.DIM}  [{elapsed:.1f}s]{C.RESET}\n")


if __name__ == "__main__":
    main()
