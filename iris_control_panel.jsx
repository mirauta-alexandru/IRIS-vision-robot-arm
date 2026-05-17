import { useState, useCallback, useMemo } from "react";

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

const DEFAULT_SEGMENTS = {
  base: 70, upper: 108, forearm: 95, wrist: 65, gripper: 55
};

const DEFAULT_SERVOS = [
  { name: "Gripper",       type: "SG90",  channel: 0, offset: 90, direction: 1,  min: 0, max: 180, role: "gripper" },
  { name: "Wrist roll",    type: "MG90S", channel: 1, offset: 90, direction: 1,  min: 0, max: 180, role: "wrist_roll" },
  { name: "Wrist pitch",   type: "MG90S", channel: 2, offset: 90, direction: 1,  min: 0, max: 180, role: "wrist_pitch" },
  { name: "Elbow",         type: "MG995", channel: 3, offset: 90, direction: 1,  min: 0, max: 180, role: "elbow" },
  { name: "Shoulder A",    type: "MG995", channel: 4, offset: 90, direction: 1,  min: 0, max: 180, role: "shoulder_a" },
  { name: "Shoulder B",    type: "MG995", channel: 5, offset: 90, direction: -1, min: 0, max: 180, role: "shoulder_b" },
  { name: "Base",          type: "MG995", channel: 6, offset: 90, direction: 1,  min: 0, max: 180, role: "base" },
];

const PRESETS = [
  { name: "Home",      angles: [90, 90, 90, 90, 90, 90, 90] },
  { name: "Forward",   xyz: [200, 0, 100] },
  { name: "Up",        xyz: [50, 0, 280] },
  { name: "Pick low",  xyz: [180, 0, 30] },
  { name: "Left",      xyz: [0, 180, 100] },
  { name: "Right",     xyz: [0, -180, 100] },
];

function solveIK(targetMM, segments, servos) {
  const s = { base: segments.base / 1000, upper: segments.upper / 1000, forearm: segments.forearm / 1000, wrist: segments.wrist / 1000, gripper: segments.gripper / 1000 };
  const tx = targetMM[0] / 1000, ty = targetMM[1] / 1000, tz = targetMM[2] / 1000;
  const baseAngle = Math.atan2(ty, tx);
  const r = Math.sqrt(tx * tx + ty * ty);
  const z = tz - s.base;
  const wristLen = s.wrist + s.gripper;
  const rw = r - wristLen * 0.3, zw = z - wristLen * 0.1;
  const d = Math.sqrt(rw * rw + zw * zw);
  const L1 = s.upper, L2 = s.forearm;
  let shoulderAngle, elbowAngle;
  if (d > L1 + L2) { shoulderAngle = Math.atan2(zw, rw); elbowAngle = 0; }
  else if (d < Math.abs(L1 - L2) + 0.001) { shoulderAngle = Math.atan2(zw, rw); elbowAngle = -Math.PI * 0.5; }
  else { const cosQ2 = Math.max(-1, Math.min(1, (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2))); elbowAngle = -Math.acos(cosQ2); const k1 = L1 + L2 * Math.cos(elbowAngle), k2 = L2 * Math.sin(elbowAngle); shoulderAngle = Math.atan2(zw, rw) - Math.atan2(k2, k1); }
  const wristPitchAngle = -(shoulderAngle + elbowAngle) * 0.5;
  const toServoDeg = (sv, rad) => Math.max(sv.min, Math.min(sv.max, Math.round((sv.offset + sv.direction * rad * RAD) * 10) / 10));
  const findServo = (role) => servos.find(sv => sv.role === role);
  const result = servos.map(sv => {
    switch (sv.role) {
      case "base": return toServoDeg(sv, baseAngle);
      case "shoulder_a": return toServoDeg(sv, shoulderAngle);
      case "shoulder_b": return toServoDeg(sv, shoulderAngle);
      case "elbow": return toServoDeg(sv, elbowAngle);
      case "wrist_pitch": return toServoDeg(sv, wristPitchAngle);
      case "wrist_roll": return sv.offset;
      case "gripper": return Math.round((sv.min + sv.max) / 2);
      default: return sv.offset;
    }
  });
  const fk = forwardK(result, segments, servos);
  const err = Math.sqrt((fk.x - targetMM[0]) ** 2 + (fk.y - targetMM[1]) ** 2 + (fk.z - targetMM[2]) ** 2);
  return { servoAngles: result, error: Math.round(err * 10) / 10 };
}

function forwardK(angles, segments, servos) {
  const s = { base: segments.base / 1000, upper: segments.upper / 1000, forearm: segments.forearm / 1000, wrist: segments.wrist / 1000, gripper: segments.gripper / 1000 };
  const getRad = (role) => { const idx = servos.findIndex(sv => sv.role === role); if (idx === -1) return 0; const sv = servos[idx]; return (angles[idx] - sv.offset) / sv.direction * DEG; };
  const baseRad = getRad("base"), shRad = getRad("shoulder_a"), elRad = getRad("elbow"), wpRad = getRad("wrist_pitch");
  const tw = s.wrist + s.gripper, elA = shRad + elRad, wA = elA + wpRad;
  const r = s.upper * Math.cos(shRad) + s.forearm * Math.cos(elA) + tw * Math.cos(wA);
  const z = s.base + s.upper * Math.sin(shRad) + s.forearm * Math.sin(elA) + tw * Math.sin(wA);
  return { x: Math.round(r * Math.cos(baseRad) * 1000 * 10) / 10, y: Math.round(r * Math.sin(baseRad) * 1000 * 10) / 10, z: Math.round(z * 1000 * 10) / 10 };
}

function getJointPositions(angles, segments, servos) {
  const s = { base: segments.base / 1000, upper: segments.upper / 1000, forearm: segments.forearm / 1000, wrist: segments.wrist / 1000, gripper: segments.gripper / 1000 };
  const getRad = (role) => { const idx = servos.findIndex(sv => sv.role === role); if (idx === -1) return 0; const sv = servos[idx]; return (angles[idx] - sv.offset) / sv.direction * DEG; };
  const b = getRad("base"), sh = getRad("shoulder_a"), el = getRad("elbow"), wp = getRad("wrist_pitch");
  const pts = [{ x: 0, y: 0, z: 0, label: "Origin" }, { x: 0, y: 0, z: s.base, label: "Base" }];
  let cr = s.upper * Math.cos(sh), cz = s.base + s.upper * Math.sin(sh);
  pts.push({ x: cr * Math.cos(b), y: cr * Math.sin(b), z: cz, label: "Shoulder" });
  const elA = sh + el; cr = s.upper * Math.cos(sh) + s.forearm * Math.cos(elA); cz = s.base + s.upper * Math.sin(sh) + s.forearm * Math.sin(elA);
  pts.push({ x: cr * Math.cos(b), y: cr * Math.sin(b), z: cz, label: "Elbow" });
  const wA = elA + wp; const rw = cr + s.wrist * Math.cos(wA), zw = cz + s.wrist * Math.sin(wA);
  pts.push({ x: rw * Math.cos(b), y: rw * Math.sin(b), z: zw, label: "Wrist" });
  const rg = rw + s.gripper * Math.cos(wA), zg = zw + s.gripper * Math.sin(wA);
  pts.push({ x: rg * Math.cos(b), y: rg * Math.sin(b), z: zg, label: "TCP" });
  return pts;
}

function ArmView({ joints, width, height, mode }) {
  const pad = 30;
  if (mode === "side") {
    const allR = joints.map(j => Math.sqrt(j.x * j.x + j.y * j.y));
    const maxR = Math.max(...allR.map(Math.abs), 0.05), maxZ = Math.max(...joints.map(j => j.z), 0.05);
    const minR = Math.min(0, ...allR.map(v => -Math.abs(v))), minZ = Math.min(0, ...joints.map(j => j.z));
    const scale = Math.min((width - pad * 2) / ((maxR - minR) || 0.1), (height - pad * 2) / ((maxZ - minZ) || 0.1)) * 0.85;
    const toSvg = (r, z) => ({ sx: pad + (r - minR) * scale, sy: height - pad - (z - minZ) * scale });
    const pts = joints.map(j => ({ ...toSvg(Math.sqrt(j.x * j.x + j.y * j.y) * Math.sign(j.x || j.y || 1), j.z), label: j.label }));
    return (<svg width={width} height={height} style={{ display: "block" }}>
      <line {...toSvg(minR, 0)} x2={toSvg(maxR, 0).sx} y2={toSvg(maxR, 0).sy} stroke="var(--color-border-tertiary)" strokeWidth="1" strokeDasharray="4 3" />
      {pts.map((p, i) => i > 0 ? <line key={`l${i}`} x1={pts[i-1].sx} y1={pts[i-1].sy} x2={p.sx} y2={p.sy} stroke={i === pts.length - 1 ? "#1D9E75" : "var(--color-text-primary)"} strokeWidth={i <= 3 ? 4 : 3} strokeLinecap="round" /> : null)}
      {pts.map((p, i) => <g key={i}><circle cx={p.sx} cy={p.sy} r={i === pts.length - 1 ? 6 : i === 0 ? 5 : 4} fill={i === pts.length - 1 ? "#1D9E75" : i === 0 ? "var(--color-text-secondary)" : "var(--color-text-primary)"} stroke="var(--color-background-primary)" strokeWidth="1.5" /><text x={p.sx} y={p.sy - 10} textAnchor="middle" style={{ fontSize: "10px", fill: "var(--color-text-secondary)" }}>{p.label}</text></g>)}
    </svg>);
  }
  const maxV = Math.max(...joints.map(j => Math.abs(j.x)), ...joints.map(j => Math.abs(j.y)), 0.05);
  const scale = Math.min(width - pad * 2, height - pad * 2) / (maxV * 2) * 0.85;
  const cx = width / 2, cy = height / 2, reach = maxV * scale;
  const pts = joints.map(j => ({ sx: cx + j.x * scale, sy: cy - j.y * scale, label: j.label }));
  return (<svg width={width} height={height} style={{ display: "block" }}>
    <circle cx={cx} cy={cy} r={reach} fill="none" stroke="var(--color-border-tertiary)" strokeWidth="0.5" strokeDasharray="3 3" />
    <line x1={cx - reach} y1={cy} x2={cx + reach} y2={cy} stroke="var(--color-border-tertiary)" strokeWidth="0.5" />
    <line x1={cx} y1={cy - reach} x2={cx} y2={cy + reach} stroke="var(--color-border-tertiary)" strokeWidth="0.5" />
    {pts.map((p, i) => i > 0 ? <line key={`l${i}`} x1={pts[i-1].sx} y1={pts[i-1].sy} x2={p.sx} y2={p.sy} stroke={i === pts.length - 1 ? "#1D9E75" : "var(--color-text-primary)"} strokeWidth={3} strokeLinecap="round" /> : null)}
    {pts.map((p, i) => <circle key={i} cx={p.sx} cy={p.sy} r={i === pts.length - 1 ? 5 : 3} fill={i === pts.length - 1 ? "#1D9E75" : "var(--color-text-primary)"} stroke="var(--color-background-primary)" strokeWidth="1" />)}
  </svg>);
}

const RC = { gripper: "#1D9E75", wrist_roll: "#5DCAA5", wrist_pitch: "#5DCAA5", elbow: "#378ADD", shoulder_a: "#7F77DD", shoulder_b: "#7F77DD", base: "#D85A30" };

export default function IRISControlPanel() {
  const [servos, setServos] = useState(DEFAULT_SERVOS);
  const [segments, setSegments] = useState(DEFAULT_SEGMENTS);
  const [angles, setAngles] = useState([90, 90, 90, 90, 90, 90, 90]);
  const [ikTarget, setIkTarget] = useState([200, 0, 100]);
  const [ikResult, setIkResult] = useState(null);
  const [tab, setTab] = useState("control");
  const [cmdLog, setCmdLog] = useState([]);
  const [mirrorLock, setMirrorLock] = useState(true);

  const joints = useMemo(() => getJointPositions(angles, segments, servos), [angles, segments, servos]);
  const tcp = useMemo(() => forwardK(angles, segments, servos), [angles, segments, servos]);

  const setAngle = useCallback((idx, val) => {
    setAngles(prev => {
      const next = [...prev]; next[idx] = Number(val);
      const sv = servos[idx];
      if (mirrorLock && (sv.role === "shoulder_a" || sv.role === "shoulder_b")) {
        const otherIdx = servos.findIndex(s => s.role === (sv.role === "shoulder_a" ? "shoulder_b" : "shoulder_a"));
        if (otherIdx !== -1) { const o = servos[otherIdx]; next[otherIdx] = Math.max(o.min, Math.min(o.max, Math.round(o.offset + o.offset - Number(val)))); }
      }
      return next;
    });
  }, [servos, mirrorLock]);

  const updateServoConfig = useCallback((idx, field, val) => { setServos(prev => { const n = [...prev]; n[idx] = { ...n[idx], [field]: Number(val) }; return n; }); }, []);
  const updateSegment = useCallback((key, val) => { setSegments(prev => ({ ...prev, [key]: Number(val) })); }, []);

  const runIK = useCallback(() => {
    const res = solveIK(ikTarget, segments, servos); setIkResult(res); setAngles(res.servoAngles);
    setCmdLog(prev => [`[IK] ${res.servoAngles.map((a, i) => `ch${servos[i].channel}:${a}`).join(" ")} err:${res.error}mm`, ...prev].slice(0, 30));
  }, [ikTarget, segments, servos]);

  const applyPreset = useCallback((p) => {
    if (p.angles) { setAngles([...p.angles]); setCmdLog(prev => [`[${p.name}]`, ...prev].slice(0, 30)); }
    else if (p.xyz) { setIkTarget([...p.xyz]); const r = solveIK(p.xyz, segments, servos); setIkResult(r); setAngles(r.servoAngles); setCmdLog(prev => [`[${p.name}] err:${r.error}mm`, ...prev].slice(0, 30)); }
  }, [segments, servos]);

  const totalReach = segments.upper + segments.forearm + segments.wrist + segments.gripper;
  const tabs = [{ id: "control", l: "Servo control" }, { id: "ik", l: "Inverse kinematics" }, { id: "calibrate", l: "Calibrate" }];

  return (
    <div style={{ padding: "1rem 0" }}>
      <h2 className="sr-only">IRIS 6-DOF robotic arm control panel</h2>
      <div style={{ display: "flex", gap: "4px", marginBottom: "1.5rem", borderBottom: "0.5px solid var(--color-border-tertiary)", paddingBottom: "8px" }}>
        {tabs.map(t => <button key={t.id} onClick={() => setTab(t.id)} style={{ padding: "6px 14px", fontSize: "13px", fontWeight: 500, background: tab === t.id ? "var(--color-background-secondary)" : "transparent", border: tab === t.id ? "0.5px solid var(--color-border-secondary)" : "0.5px solid transparent", borderRadius: "var(--border-radius-md)", cursor: "pointer", color: tab === t.id ? "var(--color-text-primary)" : "var(--color-text-secondary)" }}>{t.l}</button>)}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 310px", gap: "1.5rem" }}>
        <div>
          {tab === "control" && <div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginBottom: "1rem" }}>
              {PRESETS.map(p => <button key={p.name} onClick={() => applyPreset(p)} style={{ padding: "4px 12px", fontSize: "12px", background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-md)", color: "var(--color-text-primary)", cursor: "pointer" }}>{p.name}</button>)}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "14px", padding: "6px 10px", background: "var(--color-background-info)", borderRadius: "var(--border-radius-md)" }}>
              <label style={{ fontSize: "12px", color: "var(--color-text-info)", cursor: "pointer", display: "flex", alignItems: "center", gap: "6px" }}>
                <input type="checkbox" checked={mirrorLock} onChange={e => setMirrorLock(e.target.checked)} />
                Shoulder mirror lock (ch4 + ch5 linked)
              </label>
            </div>
            {servos.map((sv, i) => <div key={i} style={{ marginBottom: "12px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "3px" }}>
                <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: RC[sv.role], flexShrink: 0 }} />
                <span style={{ fontSize: "13px", fontWeight: 500, color: "var(--color-text-primary)", minWidth: "95px" }}>{sv.name}</span>
                <span style={{ fontSize: "10px", padding: "1px 6px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", color: "var(--color-text-secondary)" }}>{sv.type}</span>
                <span style={{ fontSize: "10px", padding: "1px 6px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>ch{sv.channel}</span>
                {sv.role === "shoulder_b" && mirrorLock && <span style={{ fontSize: "10px", color: "var(--color-text-info)" }}>linked</span>}
                <span style={{ marginLeft: "auto", fontSize: "14px", fontWeight: 500, fontFamily: "var(--font-mono)", minWidth: "45px", textAlign: "right", color: "var(--color-text-primary)" }}>{angles[i]}°</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "10px", color: "var(--color-text-tertiary)", minWidth: "24px" }}>{sv.min}°</span>
                <input type="range" min={sv.min} max={sv.max} step="1" value={angles[i]} onChange={e => setAngle(i, e.target.value)} disabled={sv.role === "shoulder_b" && mirrorLock} style={{ flex: 1, opacity: (sv.role === "shoulder_b" && mirrorLock) ? 0.4 : 1 }} />
                <span style={{ fontSize: "10px", color: "var(--color-text-tertiary)", minWidth: "28px" }}>{sv.max}°</span>
              </div>
            </div>)}
            <button onClick={() => { const cmds = angles.map((a, i) => `S${servos[i].channel}:${a}`).join("\n"); navigator.clipboard.writeText(cmds); setCmdLog(prev => ["[SERIAL] Copied", ...prev].slice(0, 30)); }} style={{ marginTop: "6px", padding: "8px 16px", fontSize: "13px", fontWeight: 500, background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-secondary)", borderRadius: "var(--border-radius-md)", color: "var(--color-text-primary)", cursor: "pointer", width: "100%" }}>Copy serial commands</button>
          </div>}

          {tab === "ik" && <div>
            <div style={{ marginBottom: "1.25rem" }}>
              {["X", "Y", "Z"].map((axis, i) => <div key={axis} style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "10px" }}>
                <label style={{ fontSize: "13px", fontWeight: 500, color: "var(--color-text-primary)", minWidth: "20px" }}>{axis}</label>
                <input type="range" min={-350} max={350} step="5" value={ikTarget[i]} onChange={e => { const n = [...ikTarget]; n[i] = Number(e.target.value); setIkTarget(n); }} style={{ flex: 1 }} />
                <input type="number" value={ikTarget[i]} onChange={e => { const n = [...ikTarget]; n[i] = Number(e.target.value); setIkTarget(n); }} style={{ width: "70px", padding: "4px 8px", fontSize: "13px", fontFamily: "var(--font-mono)", textAlign: "right", background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-md)", color: "var(--color-text-primary)" }} />
                <span style={{ fontSize: "11px", color: "var(--color-text-tertiary)" }}>mm</span>
              </div>)}
            </div>
            <button onClick={runIK} style={{ padding: "10px 20px", fontSize: "13px", fontWeight: 500, background: "#1D9E75", color: "#fff", border: "none", borderRadius: "var(--border-radius-md)", cursor: "pointer", width: "100%" }}>Solve IK & apply</button>
            {ikResult && <div style={{ marginTop: "12px", padding: "12px 14px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                <span style={{ fontSize: "12px", color: "var(--color-text-secondary)" }}>IK error</span>
                <span style={{ fontSize: "13px", fontWeight: 500, fontFamily: "var(--font-mono)", color: ikResult.error < 5 ? "#1D9E75" : ikResult.error < 15 ? "#BA7517" : "#E24B4A" }}>{ikResult.error} mm</span>
              </div>
              <div style={{ fontSize: "12px", color: "var(--color-text-secondary)", lineHeight: 2 }}>
                {ikResult.servoAngles.map((a, i) => <span key={i} style={{ marginRight: "10px", whiteSpace: "nowrap" }}><span style={{ color: RC[servos[i].role] }}>●</span> <span style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)", fontWeight: 500 }}>ch{servos[i].channel}:{a}°</span></span>)}
              </div>
            </div>}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "14px" }}>
              {PRESETS.filter(p => p.xyz).map(p => <button key={p.name} onClick={() => { setIkTarget([...p.xyz]); const r = solveIK(p.xyz, segments, servos); setIkResult(r); setAngles(r.servoAngles); }} style={{ padding: "4px 10px", fontSize: "11px", background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-md)", color: "var(--color-text-primary)", cursor: "pointer" }}>{p.name} ({p.xyz.join(",")})</button>)}
            </div>
          </div>}

          {tab === "calibrate" && <div>
            <p style={{ fontSize: "13px", color: "var(--color-text-secondary)", marginBottom: "1.25rem", lineHeight: 1.6 }}>
              Measure axis-to-axis distances with calipers. Adjust offsets so the neutral servo position matches physical zero.
            </p>
            <div style={{ marginBottom: "1.5rem" }}>
              <p style={{ fontSize: "11px", fontWeight: 500, color: "var(--color-text-tertiary)", marginBottom: "8px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Segment lengths (mm)</p>
              {Object.entries(segments).map(([key, val]) => <div key={key} style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
                <span style={{ fontSize: "13px", color: "var(--color-text-primary)", minWidth: "65px", textTransform: "capitalize" }}>{key}</span>
                <input type="range" min={20} max={200} step="1" value={val} onChange={e => updateSegment(key, e.target.value)} style={{ flex: 1 }} />
                <input type="number" value={val} onChange={e => updateSegment(key, e.target.value)} style={{ width: "55px", padding: "3px 6px", fontSize: "13px", fontFamily: "var(--font-mono)", textAlign: "right", background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-md)", color: "var(--color-text-primary)" }} />
                <span style={{ fontSize: "11px", color: "var(--color-text-tertiary)" }}>mm</span>
              </div>)}
            </div>
            <p style={{ fontSize: "11px", fontWeight: 500, color: "var(--color-text-tertiary)", marginBottom: "8px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Servo offsets & direction</p>
            {servos.map((sv, i) => <div key={i} style={{ padding: "10px 12px", marginBottom: "6px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: RC[sv.role] }} />
                <span style={{ fontSize: "13px", fontWeight: 500, color: "var(--color-text-primary)" }}>{sv.name}</span>
                <span style={{ fontSize: "10px", color: "var(--color-text-tertiary)", fontFamily: "var(--font-mono)" }}>ch{sv.channel} · {sv.type}</span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "8px" }}>
                <div>
                  <label style={{ fontSize: "10px", color: "var(--color-text-tertiary)", display: "block", marginBottom: "3px" }}>Offset (°)</label>
                  <input type="number" value={sv.offset} onChange={e => updateServoConfig(i, "offset", e.target.value)} style={{ width: "100%", padding: "4px 6px", fontSize: "13px", fontFamily: "var(--font-mono)", textAlign: "center", background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-md)", color: "var(--color-text-primary)" }} />
                </div>
                <div>
                  <label style={{ fontSize: "10px", color: "var(--color-text-tertiary)", display: "block", marginBottom: "3px" }}>Direction</label>
                  <select value={sv.direction} onChange={e => updateServoConfig(i, "direction", e.target.value)} style={{ width: "100%", padding: "4px 6px", fontSize: "13px", background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-md)", color: "var(--color-text-primary)" }}>
                    <option value={1}>Normal (+1)</option>
                    <option value={-1}>Reversed (-1)</option>
                  </select>
                </div>
                <div>
                  <label style={{ fontSize: "10px", color: "var(--color-text-tertiary)", display: "block", marginBottom: "3px" }}>Quick test</label>
                  <div style={{ display: "flex", gap: "3px" }}>
                    {[sv.min, Math.round((sv.min + sv.max) / 2), sv.max].map(a => <button key={a} onClick={() => setAngle(i, a)} style={{ flex: 1, padding: "4px 2px", fontSize: "10px", background: angles[i] === a ? "var(--color-text-primary)" : "var(--color-background-primary)", color: angles[i] === a ? "var(--color-background-primary)" : "var(--color-text-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-md)", cursor: "pointer", fontFamily: "var(--font-mono)" }}>{a}°</button>)}
                  </div>
                </div>
              </div>
            </div>)}
            <button onClick={() => { navigator.clipboard.writeText(JSON.stringify({ segments, servos: servos.map(s => ({ name: s.name, role: s.role, channel: s.channel, type: s.type, offset: s.offset, direction: s.direction, min: s.min, max: s.max })) }, null, 2)); setCmdLog(prev => ["[EXPORT] Config copied", ...prev].slice(0, 30)); }} style={{ marginTop: "10px", padding: "8px 16px", fontSize: "13px", fontWeight: 500, background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-secondary)", borderRadius: "var(--border-radius-md)", color: "var(--color-text-primary)", cursor: "pointer", width: "100%" }}>Export config to clipboard</button>
          </div>}
        </div>

        <div>
          <div style={{ padding: "10px", marginBottom: "10px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-lg)" }}>
            <p style={{ fontSize: "10px", color: "var(--color-text-tertiary)", marginBottom: "4px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Side view</p>
            <ArmView joints={joints} width={290} height={210} mode="side" />
          </div>
          <div style={{ padding: "10px", marginBottom: "10px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-lg)" }}>
            <p style={{ fontSize: "10px", color: "var(--color-text-tertiary)", marginBottom: "4px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Top view</p>
            <ArmView joints={joints} width={290} height={190} mode="top" />
          </div>
          <div style={{ padding: "10px 12px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-lg)", marginBottom: "10px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px" }}>
              {[{ l: "TCP X", v: tcp.x }, { l: "TCP Y", v: tcp.y }, { l: "TCP Z", v: tcp.z }, { l: "Reach", v: totalReach }].map(({ l, v }) => <div key={l} style={{ padding: "4px 6px" }}>
                <p style={{ fontSize: "10px", color: "var(--color-text-tertiary)", margin: "0 0 1px" }}>{l}</p>
                <p style={{ fontSize: "13px", fontWeight: 500, fontFamily: "var(--font-mono)", color: "var(--color-text-primary)", margin: 0 }}>{v} <span style={{ fontSize: "10px", fontWeight: 400, color: "var(--color-text-tertiary)" }}>mm</span></p>
              </div>)}
            </div>
          </div>
          {cmdLog.length > 0 && <div style={{ padding: "8px 10px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-lg)", maxHeight: "120px", overflowY: "auto" }}>
            <p style={{ fontSize: "10px", color: "var(--color-text-tertiary)", margin: "0 0 4px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Log</p>
            {cmdLog.map((line, i) => <p key={i} style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: i === 0 ? "var(--color-text-primary)" : "var(--color-text-tertiary)", margin: "1px 0", lineHeight: 1.5 }}>{line}</p>)}
          </div>}
        </div>
      </div>
    </div>
  );
}
