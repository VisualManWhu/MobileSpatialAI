// One virtual joystick: a ring element with a knob child. Pointer events only (Android WebView
// sends them for touch); each ring tracks a single pointerId so the two sticks work at once.
import { stickVector } from "./roam-math.js";

/**
 * Wires `ring` (position: fixed circle) and `knob` (its centred child, moved via transform).
 * onChange({ x, y }) fires on every move and with { x: 0, y: 0 } on release; the finger offset is
 * measured from the ring's centre, so the first touch anywhere in the ring already deflects.
 */
export function attachJoystick(ring, knob, onChange) {
  let active = null; // pointerId holding the stick

  const centre = () => {
    const r = ring.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2, radius: r.width / 2 };
  };
  const place = (x, y) => {
    knob.style.transform = `translate(${x}px, ${y}px)`;
  };
  const update = (ev) => {
    const c = centre();
    const s = stickVector(ev.clientX - c.x, ev.clientY - c.y, c.radius);
    place(s.knob.x, s.knob.y);
    onChange({ x: s.x, y: s.y });
  };
  const release = (ev) => {
    if (ev.pointerId !== active) return;
    active = null;
    place(0, 0);
    onChange({ x: 0, y: 0 });
  };

  ring.addEventListener("pointerdown", (ev) => {
    if (active !== null) return;
    active = ev.pointerId;
    ev.preventDefault();
    try {
      ring.setPointerCapture(ev.pointerId); // keeps the drag even when the finger leaves the ring
    } catch {
      // synthetic / already-released pointer: the stick still works, just without capture
    }
    update(ev);
  });
  ring.addEventListener("pointermove", (ev) => {
    if (ev.pointerId !== active) return;
    ev.preventDefault();
    update(ev);
  });
  ring.addEventListener("pointerup", release);
  ring.addEventListener("pointercancel", release);
}
