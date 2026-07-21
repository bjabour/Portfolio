# MRI Indicators For Alzheimer/Dementia Prediction

This note is domain context for feature engineering and model interpretation. It is not medical advice or a diagnostic protocol.

| Indicator | Why it matters | Medical terms and where to look in an axial MRI slice |
|---|---|---|
| Hippocampal / medial temporal lobe atrophy | A classic structural marker for Alzheimer-type disease. Hippocampal volume is often used in research as a diagnostic or progression biomarker. | The hippocampus is a memory-related structure deep inside the temporal lobe. In axial MRI it is usually low and medial, near the temporal horns of the lateral ventricles. In this 2D JPG dataset it may not always be visible, so treat it as interpretability context rather than a direct measured feature. |
| Entorhinal / parahippocampal and temporal cortical thinning | Early Alzheimer pathology often affects medial temporal structures, then spreads through temporal cortex. | Temporal cortical regions are parts of the temporal lobe cortex involved in memory, language, semantic knowledge, and object recognition. In axial images, look toward the lower-left and lower-right lateral brain. Thinning/atrophy can appear as less tissue and wider surrounding dark CSF spaces. |
| Ventricular enlargement and CSF-space increase | Brain tissue loss can indirectly enlarge fluid-filled spaces. This is often visible even in coarse 2D images. | Ventricles are dark, CSF-filled spaces near the center of the brain, often appearing as a dark butterfly or horn-shaped central structure. Enlargement can be approximated from central dark-region size and central intensity summaries. |
| Global cortical atrophy / gray-matter loss | More advanced dementia can produce broad tissue loss beyond the medial temporal area. | The cortex is the outer folded brain layer. In an axial slice, global atrophy can appear as wider dark grooves between folds, a thinner-looking outer brain rim, and more dark CSF around the brain surface. |
| Posterior atrophy | Some Alzheimer patterns involve posterior/parietal regions. | Posterior regions are toward the back of the brain, often parietal/occipital. Because these JPGs do not encode orientation metadata, posterior features should be used cautiously unless slice orientation is confirmed. |
| White matter hyperintensities, lacunes, infarcts, and microbleeds | These are important for vascular or mixed dementia, but usually require specific MRI sequences such as FLAIR or SWI. | White matter hyperintensities are bright patches in white matter; lacunes/infarcts are small tissue-loss or stroke lesions; microbleeds are tiny dark foci on blood-sensitive sequences. This dataset appears to be grayscale structural JPG slices, so these should not be overclaimed. |
| Radiomics / texture features | Texture and shape features can capture subtle predictive patterns beyond visual inspection. | Useful computed features include intensity histograms, local texture, edge density, ventricle/CSF proxies, brain-area proxies, regional grid summaries, and left-right asymmetry. These summarize ventricles, cortex, sulci, and temporal-region appearance. |

Feature-engineering priority for this project: central ventricle/CSF proxy, global brain/CSF proxy, regional intensity and texture summaries, edge density, asymmetry, and low-resolution pixel/PCA features.

References used in the project discussion:

- Structural MRI and hippocampal/medial temporal atrophy in Alzheimer-type disease.
- MRI visual-rating concepts: medial temporal atrophy, posterior atrophy, cortical atrophy, and vascular burden.
- Cerebrovascular MRI markers: white matter hyperintensities, infarcts, lacunes, microbleeds, and perivascular spaces.
- Radiomics and texture-feature studies for Alzheimer MRI prediction.
- Validation warnings from MRI classification work: avoid leakage, preserve held-out validation/test boundaries, and prefer patient-level splits when patient IDs exist.
