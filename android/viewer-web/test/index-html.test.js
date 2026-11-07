import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

// Runs index.html's inline (pre-bundle) error guard against a fake window.
function loadGuard({ withAndroid = true } = {}) {
  const html = readFileSync(new URL("../src/index.html", import.meta.url), "utf8");
  const inline = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  const calls = [];
  const listeners = {};
  const window = {
    addEventListener: (type, fn) => (listeners[type] = fn),
    Android: withAndroid
      ? { onLoaded: (n) => calls.push(["loaded", n]), onError: (m) => calls.push(["error", m]) }
      : undefined,
  };
  vm.runInNewContext(inline, { window, console: { error() {} } });
  return { window, calls, listeners };
}

test("a global error before load reports onError once", () => {
  const { calls, listeners } = loadGuard();
  listeners.error({ message: "SyntaxError in viewer.js" });
  listeners.unhandledrejection({ reason: new Error("wasm") });
  listeners.error({ message: "again" });
  assert.deepEqual(calls, [["error", "场景文件加载失败"]]);
});

test("errors after a successful load are not reported", () => {
  const { window, calls, listeners } = loadGuard();
  window.__r3dReport("loaded", 1234);
  listeners.error({ message: "late" });
  assert.deepEqual(calls, [["loaded", 1234]]);
});

test("works without the Android bridge (desktop debugging)", () => {
  const { listeners } = loadGuard({ withAndroid: false });
  assert.doesNotThrow(() => listeners.error({ message: "x" }));
});

test("index.html carries the roam toggle and both joystick rings (hidden until roam mode)", () => {
  const html = readFileSync(new URL("../src/index.html", import.meta.url), "utf8");
  assert.match(html, /id="roam-toggle"/);
  assert.match(html, /id="stick-move"/);
  assert.match(html, /id="stick-look"/);
  assert.match(html, /body:not\(\.roam\) \.stick \{ display: none; \}/);
  assert.match(html, />漫游</); // default label: orbit mode is active
});
