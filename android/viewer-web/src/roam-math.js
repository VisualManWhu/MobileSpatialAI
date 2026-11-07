// Pure maths for 漫游 (free-fly) mode: twin virtual joysticks -> camera yaw/pitch/position.
// No three.js / DOM, unit-tested with node:test. Frame: three.js world (+Y up, right-handed).

export const STICK_DEADZONE = 0.1; // fraction of the ring radius with no effect
export const MAX_SPEED_MPS = 1.5; // full left-stick deflection, metres per second (scenes are metric)
export const MAX_CLIMB_MPS = 1.0; // full height-rail deflection
export const MAX_TURN_RAD_PER_S = (120 * Math.PI) / 180; // full right-stick deflection
export const PITCH_LIMIT_RAD = (89 * Math.PI) / 180; // never flip over the pole

/**
 * Finger offset (dx, dy) in CSS px from the ring centre (screen y grows downward) -> stick
 * deflection { x: right +1, y: forward/up +1 } with |(x,y)| <= 1, deadzone removed and the rest
 * rescaled so the deadzone edge is 0 and the ring edge is 1; `knob` is where to draw the knob
 * centre (px, screen coords), clamped to the ring.
 */
export function stickVector(dx, dy, radius) {
  const len = Math.hypot(dx, dy);
  if (!(len > 0) || !(radius > 0)) return { x: 0, y: 0, knob: { x: 0, y: 0 } };
  const scale = len > radius ? radius / len : 1;
  const knob = { x: dx * scale, y: dy * scale };
  const frac = Math.min(1, len / radius);
  if (frac <= STICK_DEADZONE) return { x: 0, y: 0, knob };
  const mag = (frac - STICK_DEADZONE) / (1 - STICK_DEADZONE);
  return { x: (dx / len) * mag, y: (-dy / len) * mag, knob };
}

/** Unit view direction for yaw (about +Y, positive = turn left) and pitch (positive = up). */
export function forwardOf(yaw, pitch) {
  const c = Math.cos(pitch);
  return { x: -Math.sin(yaw) * c, y: Math.sin(pitch), z: -Math.cos(yaw) * c };
}

/** Unit vector to the camera's right, always level. */
export function rightOf(yaw) {
  return { x: Math.cos(yaw), y: 0, z: -Math.sin(yaw) };
}

/** Inverse of forwardOf; a (nearly) vertical direction gets yaw 0. */
export function yawPitchFromDirection(dir) {
  const len = Math.hypot(dir.x, dir.y, dir.z) || 1;
  const x = dir.x / len;
  const y = dir.y / len;
  const z = dir.z / len;
  const pitch = Math.asin(Math.max(-1, Math.min(1, y)));
  const yaw = Math.hypot(x, z) > 1e-6 ? Math.atan2(-x, -z) : 0;
  return { yaw, pitch };
}

function wrapPi(a) {
  let r = a % (2 * Math.PI);
  if (r > Math.PI) r -= 2 * Math.PI;
  if (r <= -Math.PI) r += 2 * Math.PI;
  return r;
}

/**
 * One integration step. `move` is the left stick (x strafe right, y walk forward along the LEVEL
 * heading: the pitch never changes the height, like walking), `look` the right stick (x turn
 * right, y pitch up), both from stickVector; `climb` in [-1, 1] moves straight up/down; dt in
 * seconds. Returns a new { position, yaw, pitch }; the input is not mutated.
 */
export function roamStep(state, move, look, climb, dtSec) {
  const dt = Math.max(0, dtSec || 0);
  const yaw = wrapPi(state.yaw - look.x * MAX_TURN_RAD_PER_S * dt);
  const pitch = Math.max(-PITCH_LIMIT_RAD, Math.min(PITCH_LIMIT_RAD, state.pitch + look.y * MAX_TURN_RAD_PER_S * dt));
  const f = forwardOf(yaw, 0); // level heading
  const r = rightOf(yaw);
  const d = MAX_SPEED_MPS * dt;
  const p = state.position;
  return {
    position: {
      x: p.x + (f.x * move.y + r.x * move.x) * d,
      y: p.y + (climb || 0) * MAX_CLIMB_MPS * dt,
      z: p.z + (f.z * move.y + r.z * move.x) * d,
    },
    yaw,
    pitch,
  };
}
