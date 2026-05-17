#!/usr/bin/env python3
"""
IRIS Agent v2.0 — Live Voice + Vision + Robot Control
═════════════════════════════════════════════════════

You SPEAK → Gemini HEARS + SEES camera → Gemini SPEAKS back + moves robot

Requirements:
  pip install google-genai pyaudio pynput requests --break-system-packages

Usage:
  export GEMINI_API_KEY=your_key
  python iris_bridge.py &
  python iris_agent_live.py
"""

import os, sys, json, time, base64, asyncio, threading, requests, urllib.request

# ─── CONFIG ───
BRIDGE_URL = "http://localhost:8765"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LIVE_MODEL = "gemini-3.1-flash-live-preview"
ER_MODEL = "gemini-robotics-er-1.6-preview"
MIC_RATE = 16000
SPK_RATE = 24000
MIC_CHUNK = 1600  # 100ms

class C:
    CYAN="\033[96m"; GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"
    MAGENTA="\033[95m"; BOLD="\033[1m"; DIM="\033[2m"; RESET="\033[0m"

# ══════════════════════════════════════════
#  ROBOT FUNCTIONS
# ══════════════════════════════════════════

def fn_move_arm(x, y, z):
    try:
        r = requests.post(f"{BRIDGE_URL}/ik", json={"x":x,"y":y,"z":z,"send":True}, timeout=8)
        d = r.json()
        if d.get("ok"):
            a = d.get("actual",{})
            print(f"  {C.GREEN}→ Arm ({a.get('x',x):.0f},{a.get('y',y):.0f},{a.get('z',z):.0f})mm{C.RESET}")
            return {"success":True,"actual":a,"error_mm":d.get("error_mm",0)}
        return {"success":False,"error":d.get("error","IK failed")}
    except Exception as e:
        return {"success":False,"error":str(e)}

def fn_open_gripper():
    try:
        requests.get(f"{BRIDGE_URL}/gripper?angle=0", timeout=3)
        print(f"  {C.GREEN}→ Gripper OPEN{C.RESET}")
        return {"success":True,"state":"open"}
    except Exception as e:
        return {"success":False,"error":str(e)}

def fn_close_gripper():
    try:
        requests.get(f"{BRIDGE_URL}/gripper?angle=70", timeout=3)
        print(f"  {C.GREEN}→ Gripper CLOSED{C.RESET}")
        return {"success":True,"state":"closed"}
    except Exception as e:
        return {"success":False,"error":str(e)}

def fn_go_home():
    fn_open_gripper(); time.sleep(0.3)
    return fn_move_arm(200, 0, 250)

def fn_wait_seconds(seconds):
    seconds = max(0.1, min(10, float(seconds)))
    time.sleep(seconds)
    print(f"  {C.DIM}⏳ {seconds}s{C.RESET}")
    return {"success":True,"waited":seconds}

def fn_detect_precise(query):
    """ER 1.6 object-center detection → robot coords."""
    try:
        r = requests.get(f"{BRIDGE_URL}/camera/snapshot", timeout=5)
        if r.status_code != 200:
            return {"success":False,"error":"No camera"}
        img_b64 = base64.b64encode(r.content).decode()
        img_w, img_h = 640, 480
        d = r.content; i = 2
        while i < len(d)-1:
            if d[i]==0xFF and d[i+1] in (0xC0,0xC1,0xC2):
                img_h=(d[i+5]<<8)|d[i+6]; img_w=(d[i+7]<<8)|d[i+8]; break
            elif d[i]==0xFF and d[i+1]==0xD9: break
            elif d[i]==0xFF and d[i+1] not in range(0xD0,0xD9) and d[i+1]!=0x01:
                i += 2+((d[i+2]<<8)|d[i+3])
            else: i+=1

        prompt = f"""Find: {query}
Return ONLY valid JSON, no markdown.

For each matching object, choose the point at the physical CENTER of the object for grabbing:
- Use the center of the top face / footprint, not the front edge, highlight, label, shadow, or visible side.
- For cubes/blocks, use the center of the square top face. If perspective makes the top face visible as a diamond/trapezoid, point to its geometric center.
- If you can see only part of the object, estimate the center of the whole object.
- Do not point to the closest/front edge. The robot needs the center, otherwise X will be too far forward.

Also estimate height_mm above the table. Use 25 for small cubes/blocks if unsure, 5 for flat paper/cards, 10 for small flat objects.

JSON format:
[{{"point":[y,x],"label":"<label>","height_mm":25}}]

Coordinates are normalized 0-1000 in [y,x] order."""
        body = {"contents":[{"parts":[
            {"inlineData":{"mimeType":"image/jpeg","data":img_b64}},
            {"text":prompt}
        ]}],"generationConfig":{"temperature":0.5,"thinkingConfig":{"thinkingBudget":0}}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{ER_MODEL}:generateContent?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        text = ""
        for c in result.get("candidates",[]):
            for p in c.get("content",{}).get("parts",[]):
                if "text" in p: text += p["text"]
        clean = text.strip()
        if clean.startswith("```"): clean = clean.split("\n",1)[1].rsplit("```",1)[0]
        try: raw = json.loads(clean.strip())
        except: return {"success":True,"objects":[],"raw":text[:200]}
        objects = []
        for p in raw:
            if "point" not in p: continue
            ny,nx = p["point"]
            px,py = int(nx*img_w/1000), int(ny*img_h/1000)
            height_mm = p.get("height_mm", 25)
            try:
                height_mm = float(height_mm)
            except (TypeError, ValueError):
                height_mm = 25.0
            height_mm = max(0.0, min(80.0, height_mm))
            try:
                vr = requests.post(
                    f"{BRIDGE_URL}/vision/click",
                    json={"px":px,"py":py,"z":height_mm,"send":False,"use_z_plane":True},
                    timeout=3
                )
                vd = vr.json()
                if vd.get("ok"):
                    rob = vd["robot"]
                    obj = {
                        "label":p.get("label","?"),
                        "robot":rob,
                        "pixel":{"x":px,"y":py},
                        "norm":[ny,nx],
                        "height_mm":height_mm,
                    }
                    objects.append(obj)
                    print(f"  {C.CYAN}• {obj['label']}: px({px},{py}) h={height_mm:.0f} → ({rob.get('x',0):.0f},{rob.get('y',0):.0f})mm{C.RESET}")
            except: pass
        print(f"  {C.GREEN}→ {len(objects)} objects{C.RESET}")
        return {"success":True,"objects":objects,"count":len(objects)}
    except Exception as e:
        return {"success":False,"error":str(e)}

TOOLS = {
    "move_arm": fn_move_arm, "open_gripper": fn_open_gripper,
    "close_gripper": fn_close_gripper, "go_home": fn_go_home,
    "wait_seconds": fn_wait_seconds, "detect_precise": fn_detect_precise,
}

TOOL_DECLS = [
    {"name":"move_arm","description":"Move arm to XYZ mm. X=forward(80-350), Y=left/right(-120..120), Z=height(0=table,200=safe). ALWAYS wait_seconds(3) after big moves!",
     "parameters":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}},"required":["x","y","z"]}},
    {"name":"open_gripper","description":"Open gripper instantly."},
    {"name":"close_gripper","description":"Close gripper instantly."},
    {"name":"go_home","description":"Arm to home position."},
    {"name":"wait_seconds","description":"Wait for servos. ALWAYS 3s after big moves!",
     "parameters":{"type":"object","properties":{"seconds":{"type":"number"}},"required":["seconds"]}},
    {"name":"detect_precise","description":"Gemini ER 1.6 precise pointing → robot XYZ coords. Use before grabbing.",
     "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}},
]

SYSTEM = """You are IRIS, an intelligent robot arm assistant. You see the workspace through a live camera feed and hear the user's voice.

WORKSPACE: X=80-350mm, Y=-120..120mm, Z=0(table)..250mm.
CRITICAL: Servos are SLOW. ALWAYS wait_seconds(3) after move_arm before gripping!

PICK: detect_precise → move above(Z=200) → wait(3) → open_gripper → move down(Z=0-30) → wait(3) → close_gripper → wait(1) → move up(Z=200)

You speak Romanian. Keep responses SHORT during actions. Be enthusiastic and friendly."""

# ══════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════

async def run():
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": SYSTEM,
        "tools": [{"function_declarations": TOOL_DECLS}],
    }

    # ─── PyAudio ───
    import pyaudio
    pa = pyaudio.PyAudio()
    mic = pa.open(format=pyaudio.paInt16, channels=1, rate=MIC_RATE,
                  input=True, frames_per_buffer=MIC_CHUNK)
    spk = pa.open(format=pyaudio.paInt16, channels=1, rate=SPK_RATE,
                  output=True, frames_per_buffer=2400)
    print(f"{C.GREEN}✓{C.RESET} Mic (16kHz) + Speaker (24kHz)")

    # ─── Push-to-talk ───
    from pynput import keyboard as kb
    is_talking = False
    running = True

    def on_press(key):
        nonlocal is_talking
        if key == kb.Key.space and not is_talking:
            is_talking = True
            print(f"\r  {C.RED}● REC{C.RESET} Vorbește...        ", end="", flush=True)

    def on_release(key):
        nonlocal is_talking, running
        if key == kb.Key.space and is_talking:
            is_talking = False
            print(f"\r  {C.DIM}○ Trimis{C.RESET}                  ")
        if key == kb.Key.esc:
            running = False

    kbl = kb.Listener(on_press=on_press, on_release=on_release)
    kbl.daemon = True; kbl.start()
    print(f"{C.GREEN}✓{C.RESET} Push-to-talk: hold SPACE | ESC to quit")

    # ─── Camera ───
    try:
        requests.get(f"{BRIDGE_URL}/camera/start", timeout=3)
        print(f"{C.GREEN}✓{C.RESET} Camera started")
    except:
        print(f"{C.YELLOW}⚠{C.RESET} No camera")

    print(f"\n{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}║  IRIS v2.0 — Ține SPACE și vorbește!         ║{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}╚══════════════════════════════════════════════╝{C.RESET}\n")

    async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
        print(f"{C.GREEN}✓{C.RESET} Live API connected!\n")

        # Camera thread — 1fps
        def cam_loop():
            lp = asyncio.new_event_loop()
            while running:
                try:
                    r = requests.get(f"{BRIDGE_URL}/camera/snapshot", timeout=3)
                    if r.status_code == 200:
                        lp.run_until_complete(session.send_realtime_input(
                            video={"data": base64.b64encode(r.content).decode(), "mime_type": "image/jpeg"}))
                except: pass
                time.sleep(1.0)
            lp.close()
        threading.Thread(target=cam_loop, daemon=True).start()

        # Mic thread — stream audio chunks when SPACE held
        def mic_loop():
            lp = asyncio.new_event_loop()
            while running:
                if is_talking:
                    try:
                        chunk = mic.read(MIC_CHUNK, exception_on_overflow=False)
                        lp.run_until_complete(session.send_realtime_input(
                            audio={"data": base64.b64encode(chunk).decode(),
                                   "mime_type": "audio/pcm;rate=16000"}))
                    except: pass
                else:
                    time.sleep(0.05)
            lp.close()
        threading.Thread(target=mic_loop, daemon=True).start()

        # Receive loop — audio out + tool calls
        while running:
            try:
                async for resp in session.receive():
                    if not running: break

                    # Audio → speaker
                    if resp.data is not None:
                        spk.write(resp.data)

                    # Tool calls
                    if resp.tool_call:
                        frs = []
                        for fc in resp.tool_call.function_calls:
                            nm, ar = fc.name, fc.args or {}
                            print(f"\n  {C.MAGENTA}⚡ {nm}({json.dumps(ar, ensure_ascii=False)}){C.RESET}")
                            try: res = TOOLS[nm](**ar) if nm in TOOLS else {"error":"unknown"}
                            except Exception as e: res = {"error":str(e)}
                            frs.append(types.FunctionResponse(id=fc.id, name=nm, response=res))
                        await session.send_tool_response(function_responses=frs)

                    # Turn complete
                    if resp.server_content and resp.server_content.turn_complete:
                        print(f"  {C.DIM}[ready]{C.RESET}")

            except Exception as e:
                if running: print(f"{C.RED}Error: {e}{C.RESET}")
                break

    mic.stop_stream(); mic.close()
    spk.stop_stream(); spk.close()
    pa.terminate()
    print(f"\n{C.DIM}La revedere!{C.RESET}")


def main():
    print(f"\n{C.CYAN}{C.BOLD}IRIS Agent v2.0 — Init{C.RESET}\n")
    if not GEMINI_API_KEY:
        print(f"{C.RED}✗ GEMINI_API_KEY not set!{C.RESET}"); sys.exit(1)
    print(f"{C.GREEN}✓{C.RESET} API key | Live: {LIVE_MODEL} | ER: {ER_MODEL}")
    for mod, name in [("google.genai","google-genai"),("pyaudio","pyaudio"),("pynput","pynput")]:
        try: __import__(mod); print(f"{C.GREEN}✓{C.RESET} {name}")
        except: print(f"{C.RED}✗ {name} → pip install {name} --break-system-packages{C.RESET}"); sys.exit(1)
    ok = False
    try:
        r=requests.get(f"{BRIDGE_URL}/ping",timeout=2); d=r.json()
        ok=True; print(f"{C.GREEN}✓{C.RESET} Bridge | Serial:{'✓' if d.get('serial') else '✗'} Vision:{'✓' if d.get('vision') else '✗'}")
    except: print(f"{C.RED}✗{C.RESET} Bridge not running!")
    asyncio.run(run())

if __name__ == "__main__":
    main()
