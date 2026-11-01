// Bundles src/viewer.js (three.js + Spark) into the Android assets. Run: npm ci && npm run build
// The output is committed so the Gradle build never needs Node.
import { build } from "esbuild";
import { copyFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(here, "../app/src/main/assets/viewer");

await mkdir(outDir, { recursive: true });
await build({
  entryPoints: [path.join(here, "src/viewer.js")],
  outfile: path.join(outDir, "viewer.js"),
  bundle: true,
  format: "esm",
  minify: true,
  target: ["chrome110"],
  legalComments: "eof",
  logLevel: "warning",
});
await copyFile(path.join(here, "src/index.html"), path.join(outDir, "index.html"));

// Third-party notices (both MIT) shipped next to the bundle.
const notices = [];
for (const pkg of ["@sparkjsdev/spark", "three"]) {
  const dir = path.join(here, "node_modules", pkg);
  const { version } = JSON.parse(await readFile(path.join(dir, "package.json"), "utf8"));
  const license = (await readFile(path.join(dir, "LICENSE"), "utf8")).trim();
  notices.push(`${pkg}@${version}\n\n${license}\n`);
}
const separator = `\n${"-".repeat(72)}\n\n`;
await writeFile(path.join(outDir, "THIRD_PARTY_LICENSES.txt"), notices.join(separator));

const { size } = await stat(path.join(outDir, "viewer.js"));
console.log(`viewer.js ${(size / 1024 / 1024).toFixed(2)} MiB -> ${outDir}`);
