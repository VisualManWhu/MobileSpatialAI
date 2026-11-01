import { test } from "node:test";
import assert from "node:assert/strict";
import {
  SPLAT_STRIDE,
  splatCount,
  robustBounds,
  homeView,
  easeInOutCubic,
  lerp,
  worldToThree,
} from "../src/scene-math.js";

/** Builds a .splat buffer (antimatter15 layout) from [x,y,z] positions. */
function makeSplat(positions) {
  const buf = new ArrayBuffer(positions.length * SPLAT_STRIDE);
  const dv = new DataView(buf);
  positions.forEach(([x, y, z], i) => {
    const o = i * SPLAT_STRIDE;
    dv.setFloat32(o, x, true);
    dv.setFloat32(o + 4, y, true);
    dv.setFloat32(o + 8, z, true);
    for (let k = 0; k < 3; k++) dv.setFloat32(o + 12 + 4 * k, 0.01, true);
    dv.setUint32(o + 24, 0xffffffff, true);
    dv.setUint8(o + 28, 255);
    dv.setUint8(o + 29, 128);
    dv.setUint8(o + 30, 128);
    dv.setUint8(o + 31, 128);
  });
  return buf;
}

test("stride is 32 bytes", () => {
  assert.equal(SPLAT_STRIDE, 32);
});

test("splatCount accepts whole multiples of 32", () => {
  assert.equal(splatCount(64), 2);
  assert.equal(splatCount(32 * 1000), 1000);
});

test("splatCount rejects empty, truncated and bogus sizes", () => {
  assert.throws(() => splatCount(0), /empty/);
  assert.throws(() => splatCount(33), /multiple/);
  assert.throws(() => splatCount(-32));
  assert.throws(() => splatCount(NaN));
});

test("robustBounds reads little-endian positions", () => {
  const b = robustBounds(makeSplat([[-1, -2, -3], [1, 2, 3]]), { lo: 0, hi: 1 });
  assert.deepEqual(b.min, [-1, -2, -3]);
  assert.deepEqual(b.max, [1, 2, 3]);
});

test("robustBounds trims outliers by percentile", () => {
  const pts = [];
  for (let i = 0; i <= 100; i++) pts.push([i / 100, 0, 0]); // 0..1
  pts.push([1000, 0, 0]); // far floater
  const b = robustBounds(makeSplat(pts));
  assert.ok(b.max[0] < 1.01, `max x ${b.max[0]} should ignore the floater`);
  assert.ok(b.min[0] >= 0 && b.min[0] < 0.05);
});

test("robustBounds skips non-finite positions and fails when none are finite", () => {
  const b = robustBounds(makeSplat([[NaN, 0, 0], [1, 1, 1], [Infinity, 0, 0]]), { lo: 0, hi: 1 });
  assert.deepEqual(b.min, [1, 1, 1]);
  assert.throws(() => robustBounds(makeSplat([[NaN, NaN, NaN]])), /finite/);
});

test("robustBounds samples large inputs without losing the range", () => {
  const pts = [];
  for (let i = 0; i < 200000; i++) pts.push([(i % 997) / 996, 0, 0]); // period coprime with the sample step
  const b = robustBounds(makeSplat(pts), { maxSamples: 5000, lo: 0, hi: 1 });
  assert.ok(b.min[0] < 0.01 && b.max[0] > 0.99);
});

test("homeView targets the bounds center from +Z, above, and fits the sphere", () => {
  const v = homeView({ min: [-1, -1, -1], max: [1, 1, 1] }, 60, 1);
  assert.deepEqual(v.target, { x: 0, y: 0, z: 0 });
  assert.ok(v.position.z > 0, "camera sits on the +Z side, looking toward -Z like ARCore's first frame");
  assert.ok(v.position.y > 0, "camera is elevated (world +Y is up)");
  const d = Math.hypot(v.position.x, v.position.y, v.position.z);
  const radius = Math.sqrt(3);
  assert.ok(d >= radius / Math.sin(Math.PI / 6) - 1e-9, "whole bounding sphere fits the view");
  assert.ok(v.near > 0 && v.near < v.far && v.far > d + radius);
});

test("homeView uses the narrower horizontal fov in portrait", () => {
  const wide = homeView({ min: [-1, -1, -1], max: [1, 1, 1] }, 60, 1);
  const tall = homeView({ min: [-1, -1, -1], max: [1, 1, 1] }, 60, 0.5);
  const len = (p) => Math.hypot(p.x, p.y, p.z);
  assert.ok(len(tall.position) > len(wide.position));
});

test("homeView handles a degenerate (single point) scene", () => {
  const v = homeView({ min: [2, 3, 4], max: [2, 3, 4] }, 60, 1);
  assert.deepEqual(v.target, { x: 2, y: 3, z: 4 });
  assert.ok(Number.isFinite(v.position.z) && v.position.z > 4);
});

test("easeInOutCubic is clamped, symmetric and monotone", () => {
  assert.equal(easeInOutCubic(-1), 0);
  assert.equal(easeInOutCubic(0), 0);
  assert.equal(easeInOutCubic(0.5), 0.5);
  assert.equal(easeInOutCubic(1), 1);
  assert.equal(easeInOutCubic(2), 1);
  assert.ok(Math.abs(easeInOutCubic(0.25) + easeInOutCubic(0.75) - 1) < 1e-12);
  let prev = 0;
  for (let t = 0; t <= 1; t += 0.01) {
    const e = easeInOutCubic(t);
    assert.ok(e >= prev - 1e-12);
    prev = e;
  }
  assert.ok(easeInOutCubic(0.1) < 0.1, "slow start");
});

test("lerp interpolates vectors", () => {
  assert.deepEqual(lerp({ x: 0, y: 0, z: 0 }, { x: 2, y: 4, z: -6 }, 0.5), { x: 1, y: 2, z: -3 });
});

test("worldToThree is identity (splat world = ARCore world, +Y up, same as three.js)", () => {
  assert.deepEqual(worldToThree({ x: 1, y: 2, z: 3 }), { x: 1, y: 2, z: 3 });
  assert.throws(() => worldToThree({ x: 1, y: "a", z: 3 }), /finite/);
  assert.throws(() => worldToThree(null));
});
