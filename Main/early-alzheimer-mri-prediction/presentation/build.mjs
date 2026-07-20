import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const dist = path.join(root, "dist");
const serverDir = path.join(dist, "server");

await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });

for (const entry of ["index.html", "styles.css", "script.js", "assets", ".openai"]) {
  await cp(path.join(root, entry), path.join(dist, entry), { recursive: true });
}

await mkdir(serverDir, { recursive: true });

const assetFiles = [
  ["index.html", "text/html; charset=utf-8", false],
  ["styles.css", "text/css; charset=utf-8", false],
  ["script.js", "application/javascript; charset=utf-8", false],
  ["assets/mri_mild.jpg", "image/jpeg", true],
  ["assets/mri_non_demented.jpg", "image/jpeg", true],
  ["assets/mri_very_mild.jpg", "image/jpeg", true],
  ["assets/lda_validation_projection.png", "image/png", true],
  ["assets/mlp_regional_occlusion.png", "image/png", true],
  ["assets/mlp_roc_auc.png", "image/png", true],
  ["assets/mlp_operating_points.png", "image/png", true],
  ["assets/mlp_test_error_counts.png", "image/png", true],
  ["assets/test_confusion_matrix.png", "image/png", true],
  ["assets/test_confusion_matrix_thresholded.png", "image/png", true],
  ["assets/val_confusion_matrix.png", "image/png", true]
];

const assets = {};
for (const [file, type, binary] of assetFiles) {
  const buffer = await readFile(path.join(root, file));
  assets[`/${file}`] = {
    type,
    encoding: binary ? "base64" : "utf8",
    body: binary ? buffer.toString("base64") : buffer.toString("utf8")
  };
}
assets["/"] = assets["/index.html"];

const worker = `const assets = ${JSON.stringify(assets)};

function decodeBase64(value) {
  const text = atob(value);
  const bytes = new Uint8Array(text.length);
  for (let index = 0; index < text.length; index += 1) {
    bytes[index] = text.charCodeAt(index);
  }
  return bytes;
}

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const pathname = url.pathname === "/" ? "/index.html" : url.pathname;
    const asset = assets[pathname] || assets["/index.html"];
    const body = asset.encoding === "base64" ? decodeBase64(asset.body) : asset.body;
    const cacheControl = asset.type.startsWith("text/html")
      ? "no-cache"
      : "public, max-age=31536000, immutable";

    return new Response(body, {
      headers: {
        "content-type": asset.type,
        "cache-control": cacheControl
      }
    });
  }
};
`;

await writeFile(path.join(serverDir, "index.js"), worker, "utf8");
