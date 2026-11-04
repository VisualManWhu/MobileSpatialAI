// Pure helpers for the viewer (no three.js / DOM): unit-tested with node:test.

/**
 * Bytes per Gaussian in the antimatter15 `.splat` layout written by
 * server/recon3d/recon/export_splat.py: pos f32x3, scale f32x3, rgba u8x4, rot u8x4 (wxyz).
 */
export const SPLAT_STRIDE = 32;

/** Number of Gaussians in a `.splat` of `byteLength` bytes; throws on empty / truncated files. */
export function splatCount(byteLength) {
  if (!Number.isFinite(byteLength) || byteLength < 0) throw new Error(`bad size ${byteLength}`);
  if (byteLength === 0) throw new Error("empty scene file");
  if (byteLength % SPLAT_STRIDE !== 0) {
    throw new Error(`size ${byteLength} is not a multiple of ${SPLAT_STRIDE}`);
  }
  return byteLength / SPLAT_STRIDE;
}

/**
 * Axis-aligned bounds of the Gaussian centers, trimmed to the [lo, hi] percentiles per axis so a
 * few far floaters don't blow up the home view. Large files are sampled evenly (maxSamples).
 * Returns { min: [x,y,z], max: [x,y,z] }.
 */
export function robustBounds(buffer, { lo = 0.02, hi = 0.98, maxSamples = 50000 } = {}) {
  const n = splatCount(buffer.byteLength);
  const dv = new DataView(buffer);
  const step = Math.max(1, Math.floor(n / maxSamples));
  const axes = [[], [], []];
  for (let i = 0; i < n; i += step) {
    const o = i * SPLAT_STRIDE;
    const x = dv.getFloat32(o, true);
    const y = dv.getFloat32(o + 4, true);
    const z = dv.getFloat32(o + 8, true);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    axes[0].push(x);
    axes[1].push(y);
    axes[2].push(z);
  }
  if (axes[0].length === 0) throw new Error("no finite gaussian positions");
  const min = [];
  const max = [];
  for (const values of axes) {
    const sorted = Float64Array.from(values).sort();
    const last = sorted.length - 1;
    min.push(sorted[Math.floor(lo * last)]);
    max.push(sorted[Math.ceil(hi * last)]);
  }
  return { min, max };
}

function centerAndRadius(bounds) {
  const c = [0, 1, 2].map((k) => (bounds.min[k] + bounds.max[k]) / 2);
  const half = [0, 1, 2].map((k) => (bounds.max[k] - bounds.min[k]) / 2);
  return { c, radius: Math.max(Math.hypot(half[0], half[1], half[2]), 0.1) };
}

/** True if p lies inside the bounds box expanded by `margin` (fraction of each extent). */
export function insideBounds(p, bounds, margin = 0.1) {
  return [p.x, p.y, p.z].every((v, k) => {
    const pad = (bounds.max[k] - bounds.min[k]) * margin;
    return v >= bounds.min[k] - pad && v <= bounds.max[k] + pad;
  });
}

const ORIGIN = { x: 0, y: 0, z: 0 };
const MINUS_Z = { x: 0, y: 0, z: -1 };

/**
 * Parses the page's home hint: `home=x,y,z` (camera position, scene world frame) and optionally
 * `look=dx,dy,dz` (view direction). Returns { position, look } or null when `home` is absent or
 * malformed; a malformed `look` is ignored (level -Z is used).
 */
export function parseHomeHint(params) {
  const parse = (s) => {
    if (typeof s !== "string") return null;
    const v = s.split(",").map(Number);
    return v.length === 3 && v.every(Number.isFinite) ? { x: v[0], y: v[1], z: v[2] } : null;
  };
  const position = parse(params.get("home"));
  if (!position) return null;
  return { position, look: parse(params.get("look")) };
}

/** Unit vector along `look`; level -Z when `look` is missing or (nearly) vertical. */
function viewDirection(look) {
  if (!look) return MINUS_Z;
  if (!(Math.hypot(look.x, look.z) > 1e-6)) return MINUS_Z;
  const len = Math.hypot(look.x, look.y, look.z);
  return { x: look.x / len, y: look.y / len, z: look.z / len };
}

/**
 * Home camera. The .splat carries no camera poses, so the app passes a hint when it has one:
 * the centre of the capture trajectory (mean camera position) and the first frame's view direction.
 * The camera then stands there facing that way (the pitch is kept: a floor scan looks down, as the
 * phone did); the orbit target is d metres ahead, d = the bounds centre's depth along that
 * direction (at least 0.5 m). The hint is trusted even outside the trimmed bounds: a ground-only
 * scene has nothing above the phone, so the trajectory itself lies outside the splat box.
 *
 * Without a hint the ARCore world origin stands in — it IS the device pose when the session
 * started, i.e. (almost exactly) the first captured frame's camera position, looking roughly -Z —
 * but only when it lies inside the trimmed scene bounds (expanded 10%), the normal "scanned from
 * inside the room" case; otherwise (origin outside the scene) falls back to fitView().
 */
export function homeView(bounds, fovDeg, aspect, hint = null) {
  if (!hint && !insideBounds(ORIGIN, bounds, 0.1)) return fitView(bounds, fovDeg, aspect);
  const position = hint ? hint.position : ORIGIN;
  const dir = viewDirection(hint ? hint.look : null);
  const { c, radius } = centerAndRadius(bounds);
  const rel = { x: c[0] - position.x, y: c[1] - position.y, z: c[2] - position.z };
  const d = Math.max(0.5, rel.x * dir.x + rel.y * dir.y + rel.z * dir.z);
  const reach = Math.hypot(rel.x, rel.y, rel.z) + radius;
  return {
    target: { x: position.x + dir.x * d, y: position.y + dir.y * d, z: position.z + dir.z * d },
    position: { x: position.x, y: position.y, z: position.z },
    near: Math.max(0.01, radius * 0.001),
    far: Math.max(reach, d) * 10,
  };
}

/**
 * Bounds-fit view: target = bounds centre; the camera sits on the +Z side, raised 30 degrees,
 * looking toward -Z, at a distance that fits the bounding sphere into the narrower of the
 * vertical / horizontal fov.
 */
export function fitView(bounds, fovDeg, aspect) {
  const { c, radius } = centerAndRadius(bounds);
  const vfov = (fovDeg * Math.PI) / 180;
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * aspect);
  const fov = Math.min(vfov, hfov);
  const dist = radius / Math.sin(fov / 2);
  const elev = Math.PI / 6;
  return {
    target: { x: c[0], y: c[1], z: c[2] },
    position: {
      x: c[0],
      y: c[1] + dist * Math.sin(elev),
      z: c[2] + dist * Math.cos(elev),
    },
    near: Math.max(0.01, radius * 0.001),
    far: (dist + radius) * 10,
  };
}

/**
 * Resolves/rejects like `promise`, but rejects with Error(message) if it hasn't settled after ms.
 * `timers` is injectable for tests.
 */
export function withTimeout(promise, ms, message = "timeout", timers = globalThis) {
  let id;
  const timeout = new Promise((_, reject) => {
    id = timers.setTimeout(() => reject(new Error(message)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => timers.clearTimeout(id));
}

/** Cubic ease-in-out on t clamped to [0, 1]. */
export function easeInOutCubic(t) {
  const x = Math.min(1, Math.max(0, t));
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

/** Component-wise linear interpolation of {x,y,z}. */
export function lerp(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t };
}

/**
 * Scene WORLD coordinates (same frame as the .splat positions) -> three.js world.
 *
 * Identity on purpose. The server converts only the CAMERA axes from OpenGL to OpenCV
 * (server/recon3d/capture/poses.py right-multiplies c2w), never the world frame, so the splat
 * lives in ARCore's world frame: right-handed, gravity-aligned, +Y up — exactly three.js's
 * convention. A Y flip (as the original brief assumed for a "CV world") would put the floor on top.
 */
export function worldToThree(p) {
  if (!p || ![p.x, p.y, p.z].every((v) => typeof v === "number" && Number.isFinite(v))) {
    throw new Error("expected finite {x, y, z}");
  }
  return { x: p.x, y: p.y, z: p.z };
}
