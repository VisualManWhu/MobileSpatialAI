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

/**
 * Home camera from the scene bounds (controller-approved simplification of the brief's
 * "first-frame camera": the .splat carries no camera poses).
 *
 * Target = bounds center; the camera sits on the +Z side, raised 30 degrees, looking toward -Z.
 * ARCore's world origin is the device pose at session start, whose view direction is roughly -Z,
 * so this approximates the first frame's heading. Distance fits the bounding sphere into the
 * narrower of the vertical / horizontal fov.
 */
export function homeView(bounds, fovDeg, aspect) {
  const c = [0, 1, 2].map((k) => (bounds.min[k] + bounds.max[k]) / 2);
  const half = [0, 1, 2].map((k) => (bounds.max[k] - bounds.min[k]) / 2);
  const radius = Math.max(Math.hypot(half[0], half[1], half[2]), 0.1);
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
