// Pure helpers for object highlighting (no three.js / DOM): unit-tested with node:test.
import { SPLAT_STRIDE } from "./scene-math.js";

const DIM_ALPHA = 0.15;

/** objects.bin -> Uint16Array of per-splat object ids; the count must match the loaded scene. */
export function parseObjectIds(buffer, splatCount) {
  const n = Math.floor(buffer.byteLength / 2);
  if (n !== splatCount) throw new Error(`objects.bin has ${n} ids, scene has ${splatCount} splats`);
  const dv = new DataView(buffer);
  const ids = new Uint16Array(n);
  for (let i = 0; i < n; i++) ids[i] = dv.getUint16(i * 2, true);
  return ids;
}

/** The rgba bytes (24..27 of each 32-byte record) of a .splat buffer, as one flat Uint8Array. */
export function extractRgba(splatBytes) {
  const n = Math.floor(splatBytes.byteLength / SPLAT_STRIDE);
  const src = new Uint8Array(splatBytes);
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const o = i * SPLAT_STRIDE + 24;
    out[i * 4] = src[o];
    out[i * 4 + 1] = src[o + 1];
    out[i * 4 + 2] = src[o + 2];
    out[i * 4 + 3] = src[o + 3];
  }
  return out;
}

/**
 * Per-splat rgba with every splat NOT belonging to `selected` greyed (luma) and faded to dimAlpha
 * of its opacity; the selected object's splats keep their colour. Returns a new array.
 */
export function dimRgba(rgba, ids, selected, { dimAlpha = DIM_ALPHA } = {}) {
  const out = new Uint8Array(rgba.length);
  const n = Math.min(ids.length, rgba.length / 4);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    if (ids[i] === selected) {
      out[o] = rgba[o];
      out[o + 1] = rgba[o + 1];
      out[o + 2] = rgba[o + 2];
      out[o + 3] = rgba[o + 3];
    } else {
      const luma = Math.round(0.299 * rgba[o] + 0.587 * rgba[o + 1] + 0.114 * rgba[o + 2]);
      out[o] = luma;
      out[o + 1] = luma;
      out[o + 2] = luma;
      out[o + 3] = Math.round(rgba[o + 3] * dimAlpha);
    }
  }
  return out;
}

/**
 * Where to fly to look at an object: stand d = max(1.5 * bbox diagonal, 0.8 m) away from its centre
 * along the horizontal direction from the centre towards the camera (fallback: opposite of the home
 * look direction, else +Z), 0.3 d above the centre, looking at the centre.
 */
export function flyPoseFor(center, bboxMin, bboxMax, cameraPos, homeLook) {
  const diag = Math.hypot(bboxMax.x - bboxMin.x, bboxMax.y - bboxMin.y, bboxMax.z - bboxMin.z);
  const d = Math.max(1.5 * diag, 0.8);
  let dx = cameraPos.x - center.x;
  let dz = cameraPos.z - center.z;
  let len = Math.hypot(dx, dz);
  if (!(len > 1e-3) && homeLook) {
    dx = -homeLook.x;
    dz = -homeLook.z;
    len = Math.hypot(dx, dz);
  }
  if (!(len > 1e-3)) {
    dx = 0;
    dz = 1;
    len = 1;
  }
  return {
    position: { x: center.x + (dx / len) * d, y: center.y + 0.3 * d, z: center.z + (dz / len) * d },
    lookAt: { x: center.x, y: center.y, z: center.z },
  };
}
