// Recon3D on-device 3DGS viewer: three.js + Spark, bundled into the APK (no CDN, no network).
//
// Page URL: index.html?scene=/scene/scene.splat[&home=x,y,z[&look=dx,dy,dz]]
// (served by the app's WebViewAssetLoader). `home` is the capture trajectory's centre and `look`
// the first frame's view direction, both in the scene world frame; see homeView().
// Calls back into the app via the `Android` JavascriptInterface: onLoaded(count) / onError(msgZh) /
// onRoam(on) (the app locks the screen to landscape while roaming).
//
// JS API on window (used by 第二期):
//   loadScene(url, homeHint?)           -> Promise<number>  (gaussian count)
//   flyTo({x,y,z}, {x,y,z} lookAt, ms)  -> Promise<void>    (smooth ease-in-out)
//   setCameraHome()
//   setRoam(on)                         漫游 (free-fly) mode on/off; also the page's own button
// Coordinates are the scene WORLD frame (same as .splat positions). That frame is ARCore's
// world (+Y up), identical to three.js's, so no conversion is applied — see worldToThree().

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh, SplatFileType } from "@sparkjsdev/spark";
import {
  splatCount,
  robustBounds,
  homeView,
  parseHomeHint,
  easeInOutCubic,
  lerp,
  worldToThree,
  withTimeout,
} from "./scene-math.js";
import { forwardOf, roamStep, yawPitchFromDirection } from "./roam-math.js";
import { attachJoystick } from "./joystick.js";

const MSG_LOAD_FAILED = "场景文件加载失败";
const FOV = 60;
const INIT_TIMEOUT_MS = 45000; // < the app's 60 s watchdog, so the specific error wins
const LABEL_ROAM = "漫游"; // button shows the mode you would switch TO
const LABEL_ORBIT = "环绕";
const ORBIT_RESUME_DIST_M = 2; // leaving roam: orbit target this far ahead of the camera
const MAX_STEP_S = 0.1; // a long frame (tab hidden, sort stall) must not teleport the camera

// Created in init() inside boot()'s try: a WebGL2 / wasm / worker failure must reach onError.
let renderer = null;
let scene = null;
let camera = null;
let controls = null;
let roam = null; // { yaw, pitch, move: {x,y}, look: {x,y}, lastT } while in 漫游 mode

function init() {
  renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x111111, 1);
  document.body.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(FOV, window.innerWidth / window.innerHeight, 0.01, 1000);

  // Spark re-sorts splats asynchronously after the camera moves; onDirty tells us a new frame is
  // needed once that (or a LoD update) finishes, so on-demand rendering never shows a stale sort.
  const spark = new SparkRenderer({ renderer, onDirty: () => requestRender() });
  scene.add(spark);

  // Touch: 1 finger rotate, 2 fingers dolly + pan (OrbitControls defaults), with damping.
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.minDistance = 0.1;
  controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
  controls.addEventListener("change", () => requestRender());

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    requestRender();
  });

  initRoamUi();
}

/** 漫游 overlay (index.html): toggle + sticks. Left = walk, rail = height, right = look. */
function initRoamUi() {
  const toggle = document.getElementById("roam-toggle");
  const stick = (id, apply, opts) => {
    const ring = document.getElementById(id);
    if (!ring) return;
    attachJoystick(
      ring,
      ring.querySelector(".knob"),
      (v) => {
        if (!roam) return;
        apply(v);
        requestRender();
      },
      opts,
    );
  };
  stick("stick-move", (v) => (roam.move = v));
  stick("stick-climb", (v) => (roam.climb = v.y), { vertical: true });
  stick("stick-look", (v) => (roam.look = v));
  if (toggle) toggle.addEventListener("click", () => setRoam(!roam));
}

/** Enters / leaves 漫游 mode. Entering keeps the current view; leaving orbits around a point ahead. */
function setRoam(on) {
  if (!camera || Boolean(on) === Boolean(roam)) return;
  cancelFlight();
  const toggle = document.getElementById("roam-toggle");
  if (on) {
    roam = { yaw: 0, pitch: 0, move: { x: 0, y: 0 }, look: { x: 0, y: 0 }, climb: 0, lastT: 0 };
    syncRoamFromCamera();
    controls.enabled = false;
    document.body.classList.add("roam");
    if (toggle) toggle.textContent = LABEL_ORBIT;
    notifyRoam(true);
  } else {
    const f = forwardOf(roam.yaw, roam.pitch);
    roam = null;
    controls.target.set(
      camera.position.x + f.x * ORBIT_RESUME_DIST_M,
      camera.position.y + f.y * ORBIT_RESUME_DIST_M,
      camera.position.z + f.z * ORBIT_RESUME_DIST_M,
    );
    controls.enabled = true;
    controls.update();
    document.body.classList.remove("roam");
    if (toggle) toggle.textContent = LABEL_ROAM;
    notifyRoam(false);
  }
  requestRender();
}

/** Tells the app about the mode change; older app builds without onRoam are fine. */
function notifyRoam(on) {
  const a = window.Android;
  if (a && typeof a.onRoam === "function") a.onRoam(on);
}

/** Re-derives yaw/pitch from the camera -> orbit target line (after flyTo / home while roaming). */
function syncRoamFromCamera() {
  if (!roam) return;
  const d = {
    x: controls.target.x - camera.position.x,
    y: controls.target.y - camera.position.y,
    z: controls.target.z - camera.position.z,
  };
  const yp = yawPitchFromDirection(Math.hypot(d.x, d.y, d.z) > 1e-6 ? d : forwardOf(0, 0));
  roam.yaw = yp.yaw;
  roam.pitch = yp.pitch;
  roam.lastT = 0;
}

function roamFrame(now) {
  const dt = roam.lastT ? Math.min(MAX_STEP_S, (now - roam.lastT) / 1000) : 0;
  roam.lastT = now;
  const pushing = roam.move.x || roam.move.y || roam.look.x || roam.look.y || roam.climb;
  if (!pushing) {
    roam.lastT = 0; // next push starts a fresh clock, not one from seconds ago
    return false;
  }
  const next = roamStep(
    { position: camera.position, yaw: roam.yaw, pitch: roam.pitch },
    roam.move,
    roam.look,
    roam.climb,
    dt,
  );
  roam.yaw = next.yaw;
  roam.pitch = next.pitch;
  const f = forwardOf(next.yaw, next.pitch);
  camera.position.set(next.position.x, next.position.y, next.position.z);
  // Keep the orbit target one unit ahead so leaving roam / flyTo start from a sane line of sight.
  controls.target.set(next.position.x + f.x, next.position.y + f.y, next.position.z + f.z);
  camera.lookAt(controls.target);
  return true;
}

let mesh = null;
let home = null;
let flight = null; // { from, to, fromTarget, toTarget, start, duration, resolve }
let frameQueued = false;

/** Render-on-demand: one frame per request; keeps going only while damping or a flyTo is active. */
function requestRender() {
  if (frameQueued || !renderer) return;
  frameQueued = true;
  requestAnimationFrame(frame);
}

function frame(now) {
  frameQueued = false;
  let active = false;
  if (flight) {
    const t = flight.duration > 0 ? (now - flight.start) / flight.duration : 1;
    const e = easeInOutCubic(t);
    const p = lerp(flight.from, flight.to, e);
    const q = lerp(flight.fromTarget, flight.toTarget, e);
    camera.position.set(p.x, p.y, p.z);
    controls.target.set(q.x, q.y, q.z);
    camera.lookAt(controls.target);
    if (t >= 1) {
      const done = flight.resolve;
      flight = null;
      if (roam) syncRoamFromCamera();
      else {
        controls.enabled = true;
        controls.update();
      }
      done();
    } else {
      active = true;
    }
  } else if (roam) {
    active = roamFrame(now);
  } else {
    // update() returns true while damping is still moving the camera.
    active = controls.update();
  }
  renderer.render(scene, camera);
  if (active) requestRender();
}


function vec(p) {
  return { x: p.x, y: p.y, z: p.z };
}

function cancelFlight() {
  if (!flight) return;
  const done = flight.resolve;
  flight = null;
  if (roam) syncRoamFromCamera();
  else controls.enabled = true;
  done();
}

/** Smoothly moves the camera to `pos` looking at `lookAt` (scene world coords) over durationMs. */
function flyTo(pos, lookAt, durationMs = 1000) {
  if (!camera) throw new Error("viewer not initialised");
  const to = worldToThree(pos);
  const toTarget = worldToThree(lookAt);
  cancelFlight();
  return new Promise((resolve) => {
    controls.enabled = false;
    flight = {
      from: vec(camera.position),
      to,
      fromTarget: vec(controls.target),
      toTarget,
      start: performance.now(),
      duration: Math.max(0, Number(durationMs) || 0),
      resolve,
    };
    requestRender();
  });
}

/** Jumps to the bounds-derived home view of the current scene. */
function setCameraHome() {
  if (!home) return;
  cancelFlight();
  camera.near = home.near;
  camera.far = home.far;
  camera.updateProjectionMatrix();
  controls.maxDistance = home.far * 0.5; // dollying can't push the scene past the far plane
  camera.position.set(home.position.x, home.position.y, home.position.z);
  controls.target.set(home.target.x, home.target.y, home.target.z);
  camera.lookAt(controls.target);
  if (roam) syncRoamFromCamera();
  else controls.update();
  requestRender();
}

/**
 * Fetches and shows a .splat; resolves to the gaussian count. Replaces any previous scene.
 * homeHint: optional { position, look } for the home camera (see homeView()).
 */
async function loadScene(url, homeHint = null) {
  if (!renderer) throw new Error("viewer not initialised");
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const bytes = await res.arrayBuffer();
  const count = splatCount(bytes.byteLength); // throws on empty / truncated
  const bounds = robustBounds(bytes); // computed before Spark takes the buffer

  const next = new SplatMesh({ fileBytes: bytes, fileType: SplatFileType.SPLAT, fileName: "scene.splat" });
  try {
    await withTimeout(next.initialized, INIT_TIMEOUT_MS, "SplatMesh init timeout");
  } catch (err) {
    next.dispose();
    throw err;
  }
  if (mesh) {
    scene.remove(mesh);
    mesh.dispose();
  }
  // No transform: splat world (ARCore, +Y up) == three.js world.
  mesh = next;
  scene.add(mesh);

  home = homeView(bounds, FOV, camera.aspect, homeHint);
  setCameraHome();
  const toggle = document.getElementById("roam-toggle");
  if (toggle) toggle.hidden = false; // nothing to roam through before the first scene is up
  return count;
}

/**
 * Reports the first outcome to the app exactly once. index.html defines window.__r3dReport (shared
 * with its global error / unhandledrejection handlers); the fallback covers running without it.
 */
function report(kind, arg) {
  if (typeof window.__r3dReport === "function") return window.__r3dReport(kind, arg);
  const a = window.Android;
  if (!a) return;
  if (kind === "loaded") a.onLoaded(arg);
  else a.onError(arg);
}

async function boot() {
  const params = new URLSearchParams(window.location.search);
  const url = params.get("scene");
  try {
    if (!url) throw new Error("missing ?scene=");
    init();
    const count = await loadScene(url, parseHomeHint(params));
    report("loaded", count);
  } catch (err) {
    console.error("scene load failed", err);
    report("error", MSG_LOAD_FAILED);
  }
}

window.loadScene = loadScene;
window.flyTo = flyTo;
window.setCameraHome = setCameraHome;
window.setRoam = setRoam;

boot();
