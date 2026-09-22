import { execFileSync } from "node:child_process";
import { mkdirSync, copyFileSync, writeFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
const web = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const root = path.dirname(web);
// CI uses the exported weights committed with the site, not ignored training files.
const bundledModels = process.argv.includes("--bundled-models");
execFileSync(
  "cargo",
  ["build", "--locked", "--release", "--lib", "--target", "wasm32-unknown-unknown"],
  { cwd: root, stdio: "inherit" },
);
mkdirSync(path.join(web, "public/models"), { recursive: true });
copyFileSync(
  path.join(root, "target/wasm32-unknown-unknown/release/connect_4.wasm"),
  path.join(web, "public/engine.wasm"),
);
const choices = [
  ["small-20k", "Small · 20k", "128 → 128"],
  ["large-20k", "Large · 20k", "512 → 512 → 256"],
  ["large-120k", "Large · 120k", "512 → 512 → 256"],
  ["large-1m", "Large · 1m", "512 → 512 → 256"],
];
const models = [];
for (const [id, label, architecture] of choices) {
  const destination = path.join(web, "public/models", id + ".bin");
  const source = bundledModels ? destination : path.join(root, "models", id + ".bin");
  if (!existsSync(source)) {
    if (id === "small-20k")
      throw Error(`Required model is missing: ${source}`);
    continue;
  }
  if (source !== destination) copyFileSync(source, destination);
  models.push({ id, label, architecture, url: "/models/" + id + ".bin" });
}
writeFileSync(
  path.join(web, "public/models.json"),
  JSON.stringify(models, null, 2) + "\n",
);
console.log(
  "Built browser engine and copied",
  models.length,
  "trained models.",
);
