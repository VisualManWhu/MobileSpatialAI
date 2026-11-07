import { test } from "node:test";
import assert from "node:assert/strict";
import { attachJoystick } from "../src/joystick.js";

// Minimal element stand-in: records listeners, exposes a fixed bounding box, a style bag.
function fakeElement(rect) {
  const listeners = {};
  return {
    style: {},
    captured: [],
    addEventListener: (type, fn) => (listeners[type] = fn),
    getBoundingClientRect: () => rect,
    setPointerCapture(id) {
      this.captured.push(id);
    },
    fire: (type, ev) => listeners[type]({ preventDefault() {}, ...ev }),
  };
}

function setup() {
  const ring = fakeElement({ left: 100, top: 300, width: 140, height: 140 });
  const knob = fakeElement({});
  const changes = [];
  attachJoystick(ring, knob, (v) => changes.push(v));
  return { ring, knob, changes };
}

test("dragging moves the knob and reports the stick vector; release recentres", () => {
  const { ring, knob, changes } = setup();
  ring.fire("pointerdown", { pointerId: 7, clientX: 170, clientY: 370 }); // centre
  ring.fire("pointermove", { pointerId: 7, clientX: 170, clientY: 300 }); // straight up, full
  assert.deepEqual(ring.captured, [7]);
  assert.equal(knob.style.transform, "translate(0px, -70px)");
  const last = changes[changes.length - 1];
  assert.ok(Math.abs(last.y - 1) < 1e-9 && Math.abs(last.x) < 1e-9);
  ring.fire("pointerup", { pointerId: 7 });
  assert.equal(knob.style.transform, "translate(0px, 0px)");
  assert.deepEqual(changes[changes.length - 1], { x: 0, y: 0 });
});

test("a second finger on the same ring is ignored while the first holds it", () => {
  const { ring, knob, changes } = setup();
  ring.fire("pointerdown", { pointerId: 1, clientX: 170, clientY: 370 });
  ring.fire("pointermove", { pointerId: 1, clientX: 240, clientY: 370 });
  const before = changes.length;
  ring.fire("pointerdown", { pointerId: 2, clientX: 170, clientY: 370 });
  ring.fire("pointermove", { pointerId: 2, clientX: 170, clientY: 300 });
  ring.fire("pointerup", { pointerId: 2 });
  assert.equal(changes.length, before);
  assert.equal(knob.style.transform, "translate(70px, 0px)");
  ring.fire("pointercancel", { pointerId: 1 });
  assert.deepEqual(changes[changes.length - 1], { x: 0, y: 0 });
});

test("the knob is clamped to the ring and the finger offset is measured from the ring centre", () => {
  const { ring, knob } = setup();
  ring.fire("pointerdown", { pointerId: 3, clientX: 170, clientY: 370 });
  ring.fire("pointermove", { pointerId: 3, clientX: 1000, clientY: 370 });
  assert.equal(knob.style.transform, "translate(70px, 0px)");
});
