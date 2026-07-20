import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const outputFile = path.join(root, "..", "Early-Alzheimer-Prediction.html");

let html = await readFile(path.join(root, "index.html"), "utf8");
const css = await readFile(path.join(root, "styles.css"), "utf8");
const js = await readFile(path.join(root, "script.js"), "utf8");

const images = [
  ["assets/mri_non_demented.jpg", "image/jpeg"],
  ["assets/mri_very_mild.jpg", "image/jpeg"],
  ["assets/mri_mild.jpg", "image/jpeg"],
  ["assets/lda_validation_projection.png", "image/png"],
  ["assets/mlp_regional_occlusion.png", "image/png"],
  ["assets/mlp_roc_auc.png", "image/png"],
  ["assets/mlp_operating_points.png", "image/png"],
  ["assets/mlp_test_error_counts.png", "image/png"],
  ["assets/val_confusion_matrix.png", "image/png"],
  ["assets/test_confusion_matrix.png", "image/png"],
  ["assets/test_confusion_matrix_thresholded.png", "image/png"]
];

for (const [assetPath, mimeType] of images) {
  const bytes = await readFile(path.join(root, assetPath));
  html = html.replaceAll(assetPath, `data:${mimeType};base64,${bytes.toString("base64")}`);
}

html = html.replace('<link rel="stylesheet" href="styles.css" />', `<style>\n${css}\n</style>`);
html = html.replace('<script src="script.js"></script>', `<script>\n${js}\n</script>`);

await writeFile(outputFile, html, "utf8");
console.log(outputFile);
