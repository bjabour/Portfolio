import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, "..", "..");
const presentationDir = path.join(projectRoot, "presentation");
const outputFile = path.join(projectRoot, "Early-Alzheimer-Prediction.html");

let html = await readFile(path.join(presentationDir, "index.html"), "utf8");
const css = await readFile(path.join(presentationDir, "styles.css"), "utf8");
const js = await readFile(path.join(scriptDir, "presentation.js"), "utf8");

const images = [
  ["assets/mri_non_demented.jpg", "image/jpeg"],
  ["assets/mri_very_mild.jpg", "image/jpeg"],
  ["assets/mri_mild.jpg", "image/jpeg"],
  ["assets/mri_regions.jpg", "image/jpeg"],
  ["assets/brain_mri_example.jpg", "image/jpeg"],
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
  const bytes = await readFile(path.join(presentationDir, assetPath));
  html = html.replaceAll(assetPath, `data:${mimeType};base64,${bytes.toString("base64")}`);
}

html = html.replace('<link rel="stylesheet" href="styles.css" />', `<style>\n${css}\n</style>`);
html = html.replace(
  '<script src="../scripts/presentation/presentation.js"></script>',
  `<script>\n${js}\n</script>`
);

await writeFile(outputFile, html, "utf8");
console.log(outputFile);
