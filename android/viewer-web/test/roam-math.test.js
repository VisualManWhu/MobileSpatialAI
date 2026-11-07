import { test } from "node:test";
import assert from "node:assert/strict";
import {
  STICK_DEADZONE,
  MAX_SPEED_MPS,
  MAX_CLIMB_MPS,
  MAX_TURN_RAD_PER_S,
  PITCH_LIMIT_RAD,
  stickVector,
  forwardOf,
  rightOf,
  yawPitchFromDirection,
  roamStep,
} from "../src/roam-math.js";

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const closeV = (a, b, eps = 1e-6) => {
  close(a.x, b.x, eps);
  close(a.y, b.y, eps);
  close(a.z, b.z, eps);
};

test("stick at rest gives zero vector and centred knob", () => {
  const s = stickVector(0, 0, 70);
  assert.deepEqual(s, { x: 0, y: 0, knob: { x: 0, y: 0 } });
});

test("stick inside the deadzone gives zero but the knob still follows the finger", () => {
  const s = stickVector(3, -4, 70); // 5 px of 70 = 7% < 10%
  assert.equal(s.x, 0);
  assert.equal(s.y, 0);
  assert.deepEqual(s.knob, { x: 3, y: -4 });
});

test("stick pushed straight up (screen -y) is full forward", () => {
  const s = stickVector(0, -70, 70);
  close(s.x, 0);
  close(s.y, 1);
  assert.deepEqual(s.knob, { x: 0, y: -70 });
});

test("stick beyond the ring is clamped to the ring edge", () => {
  const s = stickVector(300, 0, 70);
  close(s.x, 1);
  close(s.y, 0);
  close(s.knob.x, 70);
  close(s.knob.y, 0);
});

test("deflection is rescaled so the deadzone edge is 0 and the ring edge is 1", () => {
  const mid = (1 + STICK_DEADZONE) / 2; // halfway between deadzone and edge
  const s = stickVector(mid * 70, 0, 70);
  close(s.x, 0.5);
});

test("forward and right vectors: yaw 0 looks down -Z, right is +X; positive yaw turns left", () => {
  closeV(forwardOf(0, 0), { x: 0, y: 0, z: -1 });
  closeV(rightOf(0), { x: 1, y: 0, z: 0 });
  closeV(forwardOf(Math.PI / 2, 0), { x: -1, y: 0, z: 0 });
  closeV(rightOf(Math.PI / 2), { x: 0, y: 0, z: -1 });
  closeV(forwardOf(0, Math.PI / 2), { x: 0, y: 1, z: 0 }); // pitch up looks up
});

test("yawPitchFromDirection round-trips forwardOf", () => {
  for (const [yaw, pitch] of [[0, 0], [0.7, -0.4], [-2.5, 1.1], [3.0, 0]]) {
    const r = yawPitchFromDirection(forwardOf(yaw, pitch));
    close(Math.atan2(Math.sin(r.yaw - yaw), Math.cos(r.yaw - yaw)), 0);
    close(r.pitch, pitch);
  }
});

test("yawPitchFromDirection of a vertical look keeps a level yaw of 0", () => {
  const r = yawPitchFromDirection({ x: 0, y: -1, z: 0 });
  close(r.yaw, 0);
  close(r.pitch, -Math.PI / 2);
});

const rest = { position: { x: 1, y: 2, z: 3 }, yaw: 0, pitch: 0 };
const zero = { x: 0, y: 0 };

test("no input leaves the state unchanged", () => {
  assert.deepEqual(roamStep(rest, zero, zero, 0, 0.5), rest);
});

test("full forward for one second walks MAX_SPEED_MPS along the level heading, height unchanged", () => {
  const s = roamStep(rest, { x: 0, y: 1 }, zero, 0, 1);
  closeV(s.position, { x: 1, y: 2, z: 3 - MAX_SPEED_MPS });
  // looking 45 degrees down still walks the floor at full speed: pitch never leaks into movement
  const down = roamStep({ ...rest, pitch: -Math.PI / 4 }, { x: 0, y: 1 }, zero, 0, 1);
  closeV(down.position, { x: 1, y: 2, z: 3 - MAX_SPEED_MPS });
  // nor does a near-vertical look slow the walk down
  const up = roamStep({ ...rest, pitch: PITCH_LIMIT_RAD }, { x: 0, y: 1 }, zero, 0, 1);
  closeV(up.position, { x: 1, y: 2, z: 3 - MAX_SPEED_MPS });
});

test("walking follows the yaw: turned 90 degrees left, forward is -X", () => {
  const s = roamStep({ ...rest, yaw: Math.PI / 2 }, { x: 0, y: 1 }, zero, 0, 1);
  closeV(s.position, { x: 1 - MAX_SPEED_MPS, y: 2, z: 3 });
});

test("half right strafe moves along the camera's right axis at half speed, level", () => {
  const s = roamStep({ ...rest, pitch: -0.5 }, { x: 0.5, y: 0 }, zero, 0, 1);
  closeV(s.position, { x: 1 + MAX_SPEED_MPS / 2, y: 2, z: 3 });
});

test("climb control: +1 rises MAX_CLIMB_MPS per second, -0.5 sinks at half rate, view unchanged", () => {
  const u = roamStep({ ...rest, pitch: -0.3 }, zero, zero, 1, 1);
  closeV(u.position, { x: 1, y: 2 + MAX_CLIMB_MPS, z: 3 });
  close(u.pitch, -0.3);
  const d = roamStep(rest, zero, zero, -0.5, 2);
  closeV(d.position, { x: 1, y: 2 - MAX_CLIMB_MPS, z: 3 });
});

test("right stick: push right turns right (yaw decreases), push up pitches up", () => {
  const s = roamStep(rest, zero, { x: 1, y: 0 }, 0, 0.5);
  close(s.yaw, -MAX_TURN_RAD_PER_S * 0.5);
  const u = roamStep(rest, zero, { x: 0, y: 1 }, 0, 0.25);
  close(u.pitch, MAX_TURN_RAD_PER_S * 0.25);
  assert.deepEqual(u.position, rest.position);
});

test("pitch is clamped to +-PITCH_LIMIT_RAD", () => {
  const s = roamStep(rest, zero, { x: 0, y: 1 }, 0, 100);
  close(s.pitch, PITCH_LIMIT_RAD);
  const d = roamStep(rest, zero, { x: 0, y: -1 }, 0, 100);
  close(d.pitch, -PITCH_LIMIT_RAD);
});

test("yaw stays wrapped within (-pi, pi]", () => {
  let s = rest;
  for (let i = 0; i < 40; i++) s = roamStep(s, zero, { x: -1, y: 0 }, 0, 0.5);
  assert.ok(s.yaw > -Math.PI && s.yaw <= Math.PI, `yaw ${s.yaw}`);
});

test("constants match the design: 10% deadzone, 1.5 m/s walk, 1 m/s climb, 120 deg/s, 89 deg pitch", () => {
  assert.equal(STICK_DEADZONE, 0.1);
  assert.equal(MAX_SPEED_MPS, 1.5);
  assert.equal(MAX_CLIMB_MPS, 1.0);
  close(MAX_TURN_RAD_PER_S, (120 * Math.PI) / 180);
  close(PITCH_LIMIT_RAD, (89 * Math.PI) / 180);
});
