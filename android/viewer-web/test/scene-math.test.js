import { test } from "node:test";
import assert from "node:assert/strict";
import {
  SPLAT_STRIDE,
  splatCount,
  robustBounds,
  homeView,
  parseHomeHint,
  fitView,
  insideBounds,
  withTimeout,
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

test("fitView targets the bounds center from +Z, above, and fits the sphere", () => {
  const v = fitView({ min: [-1, -1, -1], max: [1, 1, 1] }, 60, 1);
  assert.deepEqual(v.target, { x: 0, y: 0, z: 0 });
  assert.ok(v.position.z > 0, "camera sits on the +Z side, looking toward -Z like ARCore's first frame");
  assert.ok(v.position.y > 0, "camera is elevated (world +Y is up)");
  const d = Math.hypot(v.position.x, v.position.y, v.position.z);
  const radius = Math.sqrt(3);
  assert.ok(d >= radius / Math.sin(Math.PI / 6) - 1e-9, "whole bounding sphere fits the view");
  assert.ok(v.near > 0 && v.near < v.far && v.far > d + radius);
});

test("fitView uses the narrower horizontal fov in portrait", () => {
  const wide = fitView({ min: [-1, -1, -1], max: [1, 1, 1] }, 60, 1);
  const tall = fitView({ min: [-1, -1, -1], max: [1, 1, 1] }, 60, 0.5);
  const len = (p) => Math.hypot(p.x, p.y, p.z);
  assert.ok(len(tall.position) > len(wide.position));
});

test("fitView handles a degenerate (single point) scene", () => {
  const v = fitView({ min: [2, 3, 4], max: [2, 3, 4] }, 60, 1);
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

test("insideBounds honours the 10% margin", () => {
  const b = { min: [0, 0, 0], max: [10, 10, 10] };
  assert.ok(insideBounds({ x: 5, y: 5, z: 5 }, b));
  assert.ok(insideBounds({ x: -0.9, y: 5, z: 10.9 }, b, 0.1));
  assert.ok(!insideBounds({ x: -1.1, y: 5, z: 5 }, b, 0.1));
  assert.ok(!insideBounds({ x: 5, y: 5, z: 11.5 }, b, 0.1));
});

test("homeView: origin inside the scene -> first-frame camera at origin looking level toward -Z", () => {
  const room = { min: [-2, -1.5, -4], max: [2, 1, 1] }; // centre z = -1.5
  const v = homeView(room, 60, 0.5);
  assert.deepEqual(v.position, { x: 0, y: 0, z: 0 });
  assert.deepEqual(v.target, { x: 0, y: 0, z: -1.5 }, "level pitch, d = centre depth along -Z");
  const reach = Math.hypot(0, -0.25, -1.5) + Math.hypot(2, 1.25, 2.5);
  assert.ok(v.near > 0 && v.far >= reach, "far plane covers the whole room from the origin");
});

test("homeView: target depth is clamped to >= 0.5 m", () => {
  const behind = homeView({ min: [-2, -1, -1], max: [2, 1, 3] }, 60, 1); // centre z = +1
  assert.deepEqual(behind.target, { x: 0, y: 0, z: -0.5 });
  const shallow = homeView({ min: [-1, -1, -0.4], max: [1, 1, 0.2] }, 60, 1);
  assert.deepEqual(shallow.target, { x: 0, y: 0, z: -0.5 });
});

test("homeView: origin outside the (expanded) bounds -> bounds-fit view", () => {
  const far = { min: [5, 0, -10], max: [9, 3, -6] };
  assert.deepEqual(homeView(far, 60, 1), fitView(far, 60, 1));
});

test("homeView with a hint: camera at the trajectory centre, facing the look direction", () => {
  const room = { min: [0, -1.5, -6], max: [4, 1, 0] }; // centre (2, -0.25, -3)
  const hint = { position: { x: 1, y: 0.3, z: -1 }, look: { x: 3, y: -4, z: 0 } }; // +X, pitched down
  const v = homeView(room, 60, 1, hint);
  assert.deepEqual(v.position, { x: 1, y: 0.3, z: -1 });
  // dir = (0.6, -0.8, 0); d = (1, -0.55, -2) . dir = 0.6 + 0.44 = 1.04 -> target = pos + 1.04 * dir
  assert.ok(Math.abs(v.target.x - (1 + 0.624)) < 1e-9 && Math.abs(v.target.y - (0.3 - 0.832)) < 1e-9 && v.target.z === -1);
  assert.ok(v.far >= Math.hypot(1, -0.55, -2) + Math.hypot(2, 1.25, 3));
});

test("homeView with a hint: missing or vertical look falls back to -Z; depth clamps to 0.5 m", () => {
  const room = { min: [-2, -1, -2], max: [2, 1, 2] };
  const noLook = homeView(room, 60, 1, { position: { x: 0, y: 0, z: 1 }, look: null });
  assert.deepEqual(noLook.target, { x: 0, y: 0, z: 0 }); // centre is 1 m ahead along -Z
  const down = homeView(room, 60, 1, { position: { x: 0, y: 0, z: -1.5 }, look: { x: 0, y: -1, z: 0 } });
  assert.deepEqual(down.target, { x: 0, y: 0, z: -2 }); // centre is behind -> clamped 0.5 m
});

test("homeView trusts a hint outside the trimmed bounds (ground-only scene, phone above it)", () => {
  const ground = { min: [-2, -1.4, -4], max: [2, -0.3, -1] };
  const v = homeView(ground, 60, 1, { position: { x: 0, y: 0, z: -3 }, look: { x: 0, y: -0.36, z: -0.93 } });
  assert.deepEqual(v.position, { x: 0, y: 0, z: -3 });
  assert.ok(v.target.y < 0 && v.target.z < -3, "looks down and forward from the trajectory centre");
});

test("homeView without a hint is unchanged (origin, -Z)", () => {
  const room = { min: [-2, -1.5, -4], max: [2, 1, 1] };
  assert.deepEqual(homeView(room, 60, 0.5), homeView(room, 60, 0.5, null));
});

test("parseHomeHint reads home and look, ignores a bad look, rejects a bad home", () => {
  const p = (q) => parseHomeHint(new URLSearchParams(q));
  assert.deepEqual(p("scene=a&home=1,2.5,-3&look=0,0,-1"), { position: { x: 1, y: 2.5, z: -3 }, look: { x: 0, y: 0, z: -1 } });
  assert.deepEqual(p("home=1,2,3"), { position: { x: 1, y: 2, z: 3 }, look: null });
  assert.deepEqual(p("home=1,2,3&look=x,0,0"), { position: { x: 1, y: 2, z: 3 }, look: null });
  assert.equal(p("scene=a"), null);
  assert.equal(p("home=1,2"), null);
  assert.equal(p("home=1,NaN,3"), null);
  assert.equal(p("home=1,2,Infinity"), null);
});

test("withTimeout passes through results and rejections that settle in time", async () => {
  assert.equal(await withTimeout(Promise.resolve(7), 1000), 7);
  await assert.rejects(withTimeout(Promise.reject(new Error("boom")), 1000), /boom/);
});

test("withTimeout rejects with the given message when the promise hangs", async () => {
  await assert.rejects(withTimeout(new Promise(() => {}), 10, "init timeout"), /init timeout/);
});

test("withTimeout clears its timer once settled", async () => {
  const cleared = [];
  const timers = { setTimeout: () => 42, clearTimeout: (id) => cleared.push(id) };
  await withTimeout(Promise.resolve(1), 60000, "x", timers);
  assert.deepEqual(cleared, [42]);
});
