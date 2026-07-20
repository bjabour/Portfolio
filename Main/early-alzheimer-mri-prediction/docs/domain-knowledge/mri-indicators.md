# MRI Indicators and Image Location

This note provides domain context for feature engineering and interpretation. It is not medical advice, a radiology protocol, or evidence that the model measured each structure directly.

| Indicator | Plain-language meaning | Where it appears in an axial MRI | Project treatment |
| --- | --- | --- | --- |
| Hippocampal / medial temporal atrophy | Tissue loss in memory-related structures commonly associated with Alzheimer disease. | Low and medial in the temporal lobes, near the temporal horns. It is better assessed on lower axial or coronal views. | Interpretation context only. The high axial example does not show the hippocampi reliably. |
| Temporal cortical thinning | Loss of temporal-lobe cortical tissue involved in memory, language, and semantic knowledge. | Lower-left and lower-right outer brain on an appropriately positioned slice. | Coarse lower-lateral rectangular proxies; not a temporal-lobe segmentation. |
| Ventricular enlargement / CSF increase | Expansion of central fluid spaces that can accompany surrounding tissue loss. | Dark butterfly- or horn-shaped spaces near the image center. | Central intensity, dark-pixel, texture, and pooled-pixel features. |
| Global cortical atrophy | Broad loss of outer brain tissue. | Wider dark sulci, thinner cortex, and more fluid around the outer brain surface. | Upper/lower cortical proxies, edges, brain-area proxy, and grid summaries. |
| Posterior atrophy | Tissue loss toward posterior parietal or occipital regions. | Toward the back of an orientation-confirmed axial image. | Not assigned confidently because orientation metadata are absent. |
| White-matter hyperintensity burden | Bright white-matter signal often associated with vascular or mixed pathology. | Bright patches around ventricles or deeper white matter on FLAIR-sensitive images. | Treated cautiously; sequence metadata are absent and it is not Alzheimer-specific. |
| Texture and radiomics | Numerical patterns in intensity, edges, symmetry, and local variation. | Distributed across the full image rather than one structure. | Global statistics, edge density, asymmetry, regional grids, and pooled pixels. |

## Interpretation Priority

The deterministic feature design prioritizes:

1. Central ventricle/CSF proxies.
2. Broad cortical and brain-area proxies.
3. Local intensity and texture summaries.
4. Edge density and sulcal appearance.
5. Left-right asymmetry.
6. Low-resolution spatial pixels followed by PCA.

## Regional Occlusion Result

In the final two-stage MLP, hiding the broad upper/lower cortical proxies caused the largest validation ROC-AUC decrease, followed by the central/ventricle region and then the lower-lateral proxies.

This ranking is model-specific. The rectangles overlap, replacement patches are artificial, and the images lack anatomical registration. Therefore:

- It supports the statement that the MLP used distributed cortical and central image information.
- It does not establish a causal biomarker.
- It does not identify a clinical region of disease.
- It does not substitute for hippocampal segmentation or volumetric MRI analysis.
