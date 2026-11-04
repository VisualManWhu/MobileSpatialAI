// Recon3D on-device 3DGS viewer: three.js + Spark, bundled into the APK (no CDN, no network).
//
// Page URL: index.html?scene=/scene/scene.splat[&home=x,y,z[&look=dx,dy,dz]]
// (served by the app's WebViewAssetLoader). `home` is the capture trajectory's centre and `look`
// the first frame's view direction, both in the scene world frame; see homeView().
// Calls back into the app via the `Android` JavascriptInterface: onLoaded(count) / onError(msgZh).
//
// JS API on window (used by 第二期):
//   loadScene(url, homeHint?)           -> Promise<number>  (gaussian count)
//   flyTo({x,y,z}, {x,y,z} lookAt, ms)  -> Promise<void>    (smooth ease-in-out)
//   setCameraHome()
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

const MSG_LOAD_FAILED = "场景文件加载失败";
const FOV = 60;
const INIT_TIMEOUT_MS = 45000; // < the app's 60 s watchdog, so the specific error wins

// Created in init() inside boot()'s try: a WebGL2 / wasm / worker failure must reach onError.
let renderer = null;
let scene = null;
let camera = null;
let controls = null;

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
      controls.enabled = true;
      controls.update();
      done();
    } else {
      active = true;
    }
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
  controls.enabled = true;
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
  controls.update();
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

boot();
