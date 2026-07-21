# MRI Data Setup

Raw MRI images are intentionally excluded from this portfolio repository.

To run the experiment scripts, restore the three dataset splits under this directory:

```text
data/
`-- splits/
    |-- train/train/{Non_Demented,Very_Mild_Demented,Mild_Demented}/
    |-- val/val/{Non_Demented,Very_Mild_Demented,Mild_Demented}/
    `-- test/test/{Non_Demented,Very_Mild_Demented,Mild_Demented}/
```

Expected image counts are 4,435 training, 950 validation, and 951 test images. See [Data Setup](../docs/data-setup.md) for provenance, integrity checks, and limitations.
