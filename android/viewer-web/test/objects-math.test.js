import { test } from "node:test";
import assert from "node:assert/strict";
import { parseObjectIds, extractRgba, dimRgba, flyPoseFor } from "../src/objects-math.js";

function splatBytes(rgbas) {
  const out = new Uint8Array(rgbas.length * 32);
  rgbas.forEach((c, i) => out.set(c, i * 32 + 24));
  return out.buffer;
}

test("parseObjectIds reads little-endian uint16 and rejects a count mismatch", () => {
  const ids = parseObjectIds(new Uint8Array([1, 0, 0, 0, 255, 255]).buffer, 3);
  assert.deepEqual([...ids], [1, 0, 65535]);
  assert.throws(() => parseObjectIds(new Uint8Array([1, 0]).buffer, 3), /1 ids, scene has 3/);
});

test("extractRgba pulls bytes 24..27 of every 32-byte splat", () => {
  const rgba = extractRgba(splatBytes([[10, 20, 30, 255], [1, 2, 3, 128]]));
  assert.deepEqual([...rgba], [10, 20, 30, 255, 1, 2, 3, 128]);
});

test("dimRgba greys and fades everything but the selected object", () => {
  const rgba = new Uint8Array([200, 100, 0, 255, 50, 50, 250, 200]);
  const out = dimRgba(rgba, new Uint16Array([3, 1]), 1);
  const luma = Math.round(0.299 * 200 + 0.587 * 100 + 0.114 * 0);
  assert.deepEqual([...out], [luma, luma, luma, Math.round(255 * 0.15), 50, 50, 250, 200]);
  assert.notEqual(out, rgba); // input untouched
  assert.deepEqual([...rgba.slice(0, 4)], [200, 100, 0, 255]);
});

test("flyPoseFor stands back along the camera's horizontal direction and looks at the centre", () => {
  const c = { x: 0, y: 1, z: 0 };
  const p = flyPoseFor(c, { x: -0.1, y: 0.9, z: -0.1 }, { x: 0.1, y: 1.1, z: 0.1 }, { x: 3, y: 5, z: 0 }, null);
  const d = Math.max(1.5 * Math.hypot(0.2, 0.2, 0.2), 0.8); // -> 0.8
  assert.ok(Math.abs(p.position.x - d) < 1e-9 && Math.abs(p.position.z) < 1e-9);
  assert.ok(Math.abs(p.position.y - (1 + 0.3 * d)) < 1e-9);
  assert.deepEqual(p.lookAt, c);
});

test("flyPoseFor falls back to the reverse home look when the camera is above the object", () => {
  const c = { x: 0, y: 0, z: 0 };
  const p = flyPoseFor(c, { x: -1, y: -1, z: -1 }, { x: 1, y: 1, z: 1 }, { x: 0, y: 4, z: 0 }, { x: 0, y: -0.5, z: -1 });
  const d = 1.5 * Math.sqrt(12);
  assert.ok(Math.abs(p.position.z - d) < 1e-9 && Math.abs(p.position.x) < 1e-9); // behind = +Z when look is -Z
});
