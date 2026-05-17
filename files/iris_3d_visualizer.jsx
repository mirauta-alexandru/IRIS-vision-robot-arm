<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IRIS 6-DOF — 3D Visualizer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.9/babel.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:'IBM Plex Mono',monospace;
  background:#080c14;
  color:#c8d6e5;
  min-height:100vh;
}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#0d1117}
::-webkit-scrollbar-thumb{background:#1e3a5f;border-radius:3px}
</style>
</head>
<body>
<div id="root"></div>

<script type="text/babel">
const { useState, useRef, useCallback, useEffect } = React;

/* ═══════════════════════════════════════════
   IRIS Robot Dimensions (mm)
   ═══════════════════════════════════════════ */
const DIMS = {
  base_height: 110,
  shoulder_h: 205, shoulder_v: -44.5,
  elbow: 138,
  wrist_pitch: 27,
  wrist_roll: 43,
  gripper: 97,
};

const TOTAL_REACH = DIMS.shoulder_h + DIMS.elbow + DIMS.wrist_pitch + DIMS.wrist_roll + DIMS.gripper;

/* Servo configuration */
const SERVO_LIST = [
  { key:"base",        ch:6, offset:90,  min:0,   max:180, dir:1,  label:"Base Rotation", color:"#64748b" },
  { key:"shoulder",    ch:4, offset:150, min:0,   max:150, dir:-1, label:"Shoulder",       color:"#0ea5e9" },
  { key:"elbow",       ch:3, offset:50,  min:20,  max:180, dir:-1, label:"Elbow",          color:"#10b981" },
  { key:"wrist_pitch", ch:2, offset:90,  min:0,   max:170, dir:-1, label:"Wrist Pitch",    color:"#f59e0b" },
  { key:"wrist_roll",  ch:1, offset:55,  min:0,   max:180, dir:1,  label:"Wrist Roll",     color:"#8b5cf6" },
  { key:"gripper",     ch:0, offset:60,  min:0,   max:120, dir:1,  label:"Gripper",        color:"#ef4444" },
];

const HOME = {};
SERVO_LIST.forEach(s => HOME[s.key] = s.offset);

const servoToRad = (key, deg) => {
  const s = SERVO_LIST.find(x => x.key === key);
  return ((deg - s.offset) * s.dir * Math.PI) / 180;
};

/* ═══════════════════════════════════════════
   Forward Kinematics
   ═══════════════════════════════════════════ */
function computeFK(sd) {
  const joints = [];
  let x=0, y=0, z=0, totalPitch=0;
  const baseRad = servoToRad("base", sd.base);
  const cb = Math.cos(baseRad), sb = Math.sin(baseRad);

  joints.push({x:0,y:0,z:0,label:"Base"});

  // Shoulder
  z += DIMS.base_height;
  joints.push({x:0,y:0,z,label:"Shoulder"});

  // Upper arm (with vertical offset)
  const sRad = servoToRad("shoulder", sd.shoulder);
  totalPitch += sRad;
  const elbAngle = Math.atan2(-DIMS.shoulder_v, DIMS.shoulder_h);
  const elbDist = Math.sqrt(DIMS.shoulder_h**2 + DIMS.shoulder_v**2);
  let dx = elbDist * Math.cos(totalPitch + elbAngle);
  let dz = elbDist * Math.sin(totalPitch + elbAngle);
  x += dx*cb; y += dx*sb; z += dz;
  joints.push({x,y,z,label:"Elbow"});

  // Forearm
  const eRad = servoToRad("elbow", sd.elbow);
  totalPitch += eRad;
  dx = DIMS.elbow * Math.cos(totalPitch);
  dz = DIMS.elbow * Math.sin(totalPitch);
  x += dx*cb; y += dx*sb; z += dz;
  joints.push({x,y,z,label:"Wrist Pitch"});

  // Wrist pitch
  const wpRad = servoToRad("wrist_pitch", sd.wrist_pitch);
  totalPitch += wpRad;
  dx = DIMS.wrist_pitch * Math.cos(totalPitch);
  dz = DIMS.wrist_pitch * Math.sin(totalPitch);
  x += dx*cb; y += dx*sb; z += dz;
  joints.push({x,y,z,label:"Wrist Roll"});

  // Wrist roll -> gripper axis
  dx = DIMS.wrist_roll * Math.cos(totalPitch);
  dz = DIMS.wrist_roll * Math.sin(totalPitch);
  x += dx*cb; y += dx*sb; z += dz;
  joints.push({x,y,z,label:"Gripper Axis"});

  // Gripper tip
  dx = DIMS.gripper * Math.cos(totalPitch);
  dz = DIMS.gripper * Math.sin(totalPitch);
  x += dx*cb; y += dx*sb; z += dz;
  joints.push({x,y,z,label:"EE Tip"});

  return joints;
}

/* ═══════════════════════════════════════════
   3D Projection (orbit camera)
   ═══════════════════════════════════════════ */
function project(x3,y3,z3, camH, camV, zoom, cx, cy) {
  const ch=Math.cos(camH), sh=Math.sin(camH);
  const cv=Math.cos(camV), sv=Math.sin(camV);
  const rx = x3*ch + y3*sh;
  const ry = -x3*sh*sv + y3*ch*sv + z3*cv;
  return { px: rx*zoom + cx, py: -ry*zoom + cy };
}

/* ═══════════════════════════════════════════
   Simple Numerical IK (CCD-like)
   ═══════════════════════════════════════════ */
function solveIK(targetX, targetY, targetZ, currentServos) {
  const sd = {...currentServos};
  const activeJoints = ["base","shoulder","elbow","wrist_pitch"];
  
  for(let iter=0; iter<200; iter++){
    const joints = computeFK(sd);
    const ee = joints[joints.length-1];
    const err = Math.sqrt((ee.x-targetX)**2+(ee.y-targetY)**2+(ee.z-targetZ)**2);
    if(err < 1) return sd; // <1mm

    for(const key of activeJoints){
      const s = SERVO_LIST.find(x=>x.key===key);
      const cur = sd[key];
      let bestDeg = cur, bestErr = err;
      
      for(const delta of [-2, -0.5, 0.5, 2]){
        const test = Math.max(s.min, Math.min(s.max, cur+delta));
        sd[key] = test;
        const tj = computeFK(sd);
        const te = tj[tj.length-1];
        const te2 = Math.sqrt((te.x-targetX)**2+(te.y-targetY)**2+(te.z-targetZ)**2);
        if(te2 < bestErr){ bestErr=te2; bestDeg=test; }
      }
      sd[key] = bestDeg;
    }
  }
  return sd;
}

/* ═══════════════════════════════════════════
   Ghost trail for animation
   ═══════════════════════════════════════════ */

/* ═══════════════════════════════════════════
   Main App
   ═══════════════════════════════════════════ */
function App() {
  const [servoDeg, setServoDeg] = useState({...HOME});
  const [camH, setCamH] = useState(-0.6);
  const [camV, setCamV] = useState(0.75);
  const [zoom, setZoom] = useState(0.7);
  const [dragging, setDragging] = useState(false);
  const lastM = useRef({x:0,y:0});
  const [ikTarget, setIkTarget] = useState({x:300, y:0, z:200});
  const [showIkTarget, setShowIkTarget] = useState(false);
  const [ikError, setIkError] = useState(null);

  const W = 750, H = 550, CX = W/2, CY = H*0.7;

  const handleServo = (key, val) => setServoDeg(p => ({...p, [key]:Number(val)}));
  const resetHome = () => setServoDeg({...HOME});

  const joints = computeFK(servoDeg);
  const ee = joints[joints.length-1];
  const pts = joints.map(j => project(j.x,j.y,j.z, camH,camV,zoom, CX,CY));

  // Grid
  const gridLines = [];
  for(let i=-400;i<=400;i+=100){
    const a=project(i,-400,0,camH,camV,zoom,CX,CY), b=project(i,400,0,camH,camV,zoom,CX,CY);
    gridLines.push({x1:a.px,y1:a.py,x2:b.px,y2:b.py});
    const c=project(-400,i,0,camH,camV,zoom,CX,CY), d=project(400,i,0,camH,camV,zoom,CX,CY);
    gridLines.push({x1:c.px,y1:c.py,x2:d.px,y2:d.py});
  }

  // Axes
  const o=project(0,0,0,camH,camV,zoom,CX,CY);
  const axX=project(60,0,0,camH,camV,zoom,CX,CY);
  const axY=project(0,60,0,camH,camV,zoom,CX,CY);
  const axZ=project(0,0,60,camH,camV,zoom,CX,CY);

  // IK target projected
  const ikPt = showIkTarget ? project(ikTarget.x,ikTarget.y,ikTarget.z,camH,camV,zoom,CX,CY) : null;

  // Workspace sphere (approximate)
  const reach = TOTAL_REACH;

  const onDown = (e) => {
    setDragging(true);
    const p = e.touches ? e.touches[0] : e;
    lastM.current = {x:p.clientX, y:p.clientY};
  };
  const onMove = (e) => {
    if(!dragging) return;
    const p = e.touches ? e.touches[0] : e;
    const dx = p.clientX - lastM.current.x;
    const dy = p.clientY - lastM.current.y;
    setCamH(h => h + dx*0.008);
    setCamV(v => Math.max(0.05, Math.min(1.5, v - dy*0.008)));
    lastM.current = {x:p.clientX, y:p.clientY};
  };
  const onUp = () => setDragging(false);
  const onWheel = (e) => {
    e.preventDefault();
    setZoom(z => Math.max(0.2, Math.min(2, z - e.deltaY*0.001)));
  };

  const runIK = () => {
    const result = solveIK(ikTarget.x, ikTarget.y, ikTarget.z, servoDeg);
    setServoDeg(result);
    setShowIkTarget(true);
    const fk = computeFK(result);
    const tip = fk[fk.length-1];
    const err = Math.sqrt((tip.x-ikTarget.x)**2+(tip.y-ikTarget.y)**2+(tip.z-ikTarget.z)**2);
    setIkError(err);
  };

  const linkColors = ["#334155","#0ea5e9","#10b981","#f59e0b","#8b5cf6","#7c3aed","#ef4444"];
  const jointColors = ["#64748b","#0ea5e9","#10b981","#f59e0b","#8b5cf6","#7c3aed","#ef4444"];

  return (
    <div style={{display:"flex",gap:16,padding:16,minHeight:"100vh",background:"#080c14",flexWrap:"wrap"}}>
      
      {/* ─── 3D VIEWPORT ─── */}
      <div style={{flex:"1 1 500px",minWidth:400}}>
        <div style={{background:"#0d1117",borderRadius:16,border:"1px solid #1b2838",overflow:"hidden",boxShadow:"0 0 60px rgba(14,165,233,0.05)"}}>
          <div style={{padding:"14px 20px",borderBottom:"1px solid #1b2838",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
            <div>
              <span style={{fontFamily:"Outfit",fontSize:18,fontWeight:800,color:"#0ea5e9",letterSpacing:2}}>IRIS</span>
              <span style={{fontSize:11,color:"#3b5068",marginLeft:10}}>6-DOF ROBOT ARM</span>
            </div>
            <span style={{fontSize:10,color:"#2a3f55"}}>DRAG TO ORBIT • SCROLL TO ZOOM</span>
          </div>
          
          <svg
            width={W} height={H}
            viewBox={`0 0 ${W} ${H}`}
            style={{width:"100%",height:"auto",cursor:dragging?"grabbing":"grab",display:"block"}}
            onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
            onTouchStart={onDown} onTouchMove={onMove} onTouchEnd={onUp}
            onWheel={onWheel}
          >
            <defs>
              <radialGradient id="bg" cx="50%" cy="60%">
                <stop offset="0%" stopColor="#111927"/>
                <stop offset="100%" stopColor="#080c14"/>
              </radialGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
              <filter id="glow2">
                <feGaussianBlur stdDeviation="6" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
            </defs>

            <rect width={W} height={H} fill="url(#bg)"/>

            {/* Grid */}
            {gridLines.map((l,i) => (
              <line key={i} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} stroke="#111927" strokeWidth={0.5}/>
            ))}

            {/* Axes */}
            <line x1={o.px} y1={o.py} x2={axX.px} y2={axX.py} stroke="#ef4444" strokeWidth={1.5} opacity={0.7}/>
            <text x={axX.px+4} y={axX.py-2} fill="#ef4444" fontSize={10} fontWeight={700} opacity={0.7}>X</text>
            <line x1={o.px} y1={o.py} x2={axY.px} y2={axY.py} stroke="#22c55e" strokeWidth={1.5} opacity={0.7}/>
            <text x={axY.px+4} y={axY.py-2} fill="#22c55e" fontSize={10} fontWeight={700} opacity={0.7}>Y</text>
            <line x1={o.px} y1={o.py} x2={axZ.px} y2={axZ.py} stroke="#3b82f6" strokeWidth={1.5} opacity={0.7}/>
            <text x={axZ.px+4} y={axZ.py-2} fill="#3b82f6" fontSize={10} fontWeight={700} opacity={0.7}>Z</text>

            {/* Shadow on ground plane */}
            {pts.slice(0,-1).map((p,i) => {
              const j1 = joints[i], j2 = joints[i+1];
              const s1 = project(j1.x,j1.y,0,camH,camV,zoom,CX,CY);
              const s2 = project(j2.x,j2.y,0,camH,camV,zoom,CX,CY);
              return <line key={`sh-${i}`} x1={s1.px} y1={s1.py} x2={s2.px} y2={s2.py} stroke="#0ea5e9" strokeWidth={1} opacity={0.08}/>;
            })}

            {/* Links */}
            {pts.slice(0,-1).map((p,i) => {
              const next = pts[i+1];
              const thick = Math.max(2.5, 9 - i*1.2);
              return (
                <g key={`link-${i}`}>
                  <line x1={p.px} y1={p.py} x2={next.px} y2={next.py}
                    stroke={linkColors[i]} strokeWidth={thick+2} strokeLinecap="round" opacity={0.3}/>
                  <line x1={p.px} y1={p.py} x2={next.px} y2={next.py}
                    stroke={linkColors[i]} strokeWidth={thick} strokeLinecap="round" opacity={0.9}
                    filter="url(#glow)"/>
                </g>
              );
            })}

            {/* Joints */}
            {pts.map((p,i) => {
              const r = i===pts.length-1 ? 5 : Math.max(3.5, 8 - i*0.8);
              return (
                <g key={`j-${i}`}>
                  <circle cx={p.px} cy={p.py} r={r+2} fill={jointColors[i]} opacity={0.15} filter="url(#glow2)"/>
                  <circle cx={p.px} cy={p.py} r={r} fill={jointColors[i]} stroke="#080c14" strokeWidth={1.5}/>
                  <text x={p.px+r+6} y={p.py-r-2} fill="#3b5068" fontSize={8.5} fontWeight={500}>{joints[i].label}</text>
                </g>
              );
            })}

            {/* EE crosshair */}
            {(() => {
              const ep = pts[pts.length-1];
              return (
                <g filter="url(#glow)">
                  <circle cx={ep.px} cy={ep.py} r={14} fill="none" stroke="#ef4444" strokeWidth={1} strokeDasharray="4,3" opacity={0.6}/>
                  <line x1={ep.px-18} y1={ep.py} x2={ep.px+18} y2={ep.py} stroke="#ef4444" strokeWidth={0.6} opacity={0.5}/>
                  <line x1={ep.px} y1={ep.py-18} x2={ep.px} y2={ep.py+18} stroke="#ef4444" strokeWidth={0.6} opacity={0.5}/>
                </g>
              );
            })()}

            {/* IK target marker */}
            {ikPt && (
              <g>
                <circle cx={ikPt.px} cy={ikPt.py} r={8} fill="none" stroke="#fbbf24" strokeWidth={2} strokeDasharray="3,2"/>
                <circle cx={ikPt.px} cy={ikPt.py} r={3} fill="#fbbf24"/>
                <text x={ikPt.px+12} y={ikPt.py+4} fill="#fbbf24" fontSize={9} fontWeight={600}>TARGET</text>
              </g>
            )}

            {/* Info bar */}
            <text x={16} y={H-14} fill="#2a3f55" fontSize={10} fontFamily="IBM Plex Mono">
              EE: ({ee.x.toFixed(1)}, {ee.y.toFixed(1)}, {ee.z.toFixed(1)}) mm   |   Reach: {Math.sqrt(ee.x**2+ee.y**2+ee.z**2).toFixed(1)} mm
            </text>
          </svg>
        </div>
      </div>

      {/* ─── CONTROLS PANEL ─── */}
      <div style={{flex:"0 0 310px",display:"flex",flexDirection:"column",gap:12,maxHeight:"100vh",overflowY:"auto"}}>
        
        {/* Servo Sliders */}
        <div style={{background:"#0d1117",borderRadius:16,border:"1px solid #1b2838",padding:18}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:14}}>
            <span style={{fontFamily:"Outfit",fontSize:14,fontWeight:700,color:"#0ea5e9",letterSpacing:1}}>SERVO CONTROL</span>
            <button onClick={resetHome} style={{
              background:"#1b2838",border:"1px solid #2a3f55",color:"#3b5068",borderRadius:6,
              padding:"5px 12px",fontSize:10,cursor:"pointer",fontFamily:"IBM Plex Mono",fontWeight:600,
              transition:"all 0.2s"
            }}
            onMouseEnter={e=>{e.target.style.background="#2a3f55";e.target.style.color="#c8d6e5"}}
            onMouseLeave={e=>{e.target.style.background="#1b2838";e.target.style.color="#3b5068"}}
            >HOME</button>
          </div>

          {SERVO_LIST.map(s => (
            <div key={s.key} style={{marginBottom:12}}>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:11,marginBottom:3}}>
                <span style={{color:"#3b5068"}}>
                  <span style={{color:s.color,fontWeight:700}}>CH{s.ch}</span> {s.label}
                </span>
                <span style={{color:"#c8d6e5",fontWeight:600,fontSize:13}}>{servoDeg[s.key]}°</span>
              </div>
              <input type="range" min={s.min} max={s.max} value={servoDeg[s.key]}
                onChange={e => handleServo(s.key, e.target.value)}
                style={{width:"100%",accentColor:s.color,height:4}}/>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:8,color:"#1b2838",marginTop:1}}>
                <span>{s.min}°</span>
                <span style={{color:"#2a3f55"}}>offset {s.offset}°</span>
                <span>{s.max}°</span>
              </div>
            </div>
          ))}

          {/* CH5 mirror */}
          <div style={{background:"#111927",borderRadius:8,padding:"8px 12px",fontSize:11,color:"#3b5068",marginTop:4}}>
            CH5 Shoulder B (mirror): <span style={{color:"#c8d6e5",fontWeight:600,fontSize:13}}>{180-servoDeg.shoulder}°</span>
          </div>
        </div>

        {/* End Effector */}
        <div style={{background:"#0d1117",borderRadius:16,border:"1px solid #1b2838",padding:18}}>
          <span style={{fontFamily:"Outfit",fontSize:14,fontWeight:700,color:"#0ea5e9",letterSpacing:1,display:"block",marginBottom:10}}>END EFFECTOR</span>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8}}>
            {[["X",ee.x,"#ef4444"],["Y",ee.y,"#22c55e"],["Z",ee.z,"#3b82f6"]].map(([ax,val,col]) => (
              <div key={ax} style={{textAlign:"center",background:"#111927",borderRadius:10,padding:"10px 4px",border:"1px solid #1b2838"}}>
                <div style={{fontSize:10,color:col,fontWeight:700}}>{ax}</div>
                <div style={{fontSize:16,fontWeight:700,color:"#c8d6e5",margin:"2px 0"}}>{val.toFixed(1)}</div>
                <div style={{fontSize:8,color:"#2a3f55"}}>mm</div>
              </div>
            ))}
          </div>
          <div style={{marginTop:8,textAlign:"center",background:"#111927",borderRadius:10,padding:"8px 4px",border:"1px solid #1b2838"}}>
            <span style={{fontSize:10,color:"#3b5068"}}>Reach </span>
            <span style={{fontSize:15,fontWeight:700,color:"#fbbf24"}}>{Math.sqrt(ee.x**2+ee.y**2+ee.z**2).toFixed(1)}</span>
            <span style={{fontSize:9,color:"#3b5068"}}> mm</span>
          </div>
        </div>

        {/* IK Solver */}
        <div style={{background:"#0d1117",borderRadius:16,border:"1px solid #1b2838",padding:18}}>
          <span style={{fontFamily:"Outfit",fontSize:14,fontWeight:700,color:"#fbbf24",letterSpacing:1,display:"block",marginBottom:10}}>IK SOLVER</span>
          <div style={{display:"flex",flexDirection:"column",gap:6}}>
            {[["X",ikTarget.x,"x"],["Y",ikTarget.y,"y"],["Z",ikTarget.z,"z"]].map(([label,val,k]) => (
              <div key={k} style={{display:"flex",alignItems:"center",gap:8}}>
                <label style={{fontSize:11,color:"#3b5068",width:20,fontWeight:700}}>{label}</label>
                <input type="number" value={val}
                  onChange={e => setIkTarget(p => ({...p,[k]:Number(e.target.value)}))}
                  style={{
                    background:"#111927",border:"1px solid #1b2838",color:"#c8d6e5",
                    borderRadius:8,padding:"6px 10px",fontSize:12,fontFamily:"IBM Plex Mono",
                    width:90,outline:"none"
                  }}/>
                <span style={{fontSize:9,color:"#2a3f55"}}>mm</span>
              </div>
            ))}
          </div>
          <button onClick={runIK} style={{
            background:"linear-gradient(135deg,#0ea5e9,#0284c7)",border:"none",color:"#080c14",
            borderRadius:10,padding:"10px 16px",fontSize:12,cursor:"pointer",fontFamily:"Outfit",
            fontWeight:700,width:"100%",marginTop:12,letterSpacing:1,transition:"transform 0.15s"
          }}
          onMouseEnter={e=>e.target.style.transform="scale(1.02)"}
          onMouseLeave={e=>e.target.style.transform="scale(1)"}
          >SOLVE IK</button>
          {ikError !== null && (
            <div style={{fontSize:10,color:ikError<5?"#10b981":"#f59e0b",marginTop:8,textAlign:"center"}}>
              Error: {ikError.toFixed(2)} mm {ikError<5?"✓":"⚠ may be out of reach"}
            </div>
          )}
        </div>

        {/* ESP32 Output */}
        <div style={{background:"#0d1117",borderRadius:16,border:"1px solid #1b2838",padding:18}}>
          <span style={{fontFamily:"Outfit",fontSize:14,fontWeight:700,color:"#10b981",letterSpacing:1,display:"block",marginBottom:10}}>ESP32 OUTPUT</span>
          <div style={{background:"#111927",borderRadius:8,padding:10,fontFamily:"IBM Plex Mono",fontSize:11,color:"#3b5068",lineHeight:1.8}}>
            <div><span style={{color:"#10b981"}}>pca.</span>setPWM(<span style={{color:"#fbbf24"}}>6</span>, <span style={{color:"#c8d6e5"}}>{servoDeg.base}</span>); <span style={{color:"#1b2838"}}>// base</span></div>
            <div><span style={{color:"#10b981"}}>pca.</span>setPWM(<span style={{color:"#fbbf24"}}>4</span>, <span style={{color:"#c8d6e5"}}>{servoDeg.shoulder}</span>); <span style={{color:"#1b2838"}}>// shoulder A</span></div>
            <div><span style={{color:"#10b981"}}>pca.</span>setPWM(<span style={{color:"#fbbf24"}}>5</span>, <span style={{color:"#c8d6e5"}}>{180-servoDeg.shoulder}</span>); <span style={{color:"#1b2838"}}>// shoulder B</span></div>
            <div><span style={{color:"#10b981"}}>pca.</span>setPWM(<span style={{color:"#fbbf24"}}>3</span>, <span style={{color:"#c8d6e5"}}>{servoDeg.elbow}</span>); <span style={{color:"#1b2838"}}>// elbow</span></div>
            <div><span style={{color:"#10b981"}}>pca.</span>setPWM(<span style={{color:"#fbbf24"}}>2</span>, <span style={{color:"#c8d6e5"}}>{servoDeg.wrist_pitch}</span>); <span style={{color:"#1b2838"}}>// wrist pitch</span></div>
            <div><span style={{color:"#10b981"}}>pca.</span>setPWM(<span style={{color:"#fbbf24"}}>1</span>, <span style={{color:"#c8d6e5"}}>{servoDeg.wrist_roll}</span>); <span style={{color:"#1b2838"}}>// wrist roll</span></div>
            <div><span style={{color:"#10b981"}}>pca.</span>setPWM(<span style={{color:"#fbbf24"}}>0</span>, <span style={{color:"#c8d6e5"}}>{servoDeg.gripper}</span>); <span style={{color:"#1b2838"}}>// gripper</span></div>
          </div>
        </div>

      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
</script>
</body>
</html>
