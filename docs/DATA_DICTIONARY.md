# Data Dictionary

The checked-in authoritative schema is `src/hcc_survival/schemas.py`. It defines the 49
accepted feature names, labels, groups, data types, category codes, units, and broad
non-diagnostic plausibility bounds. `hcc_survival.data.download_dataset()` validates incoming
data against that schema before writing a local cache.

| Group | Features |
| --- | --- |
| Demographics | `Gender`, `Age` |
| Symptoms and clinical status | `Symptoms`, `Varices`, `Splenomegaly`, `PHT`, `PS`, `Encephalopathy`, `Ascites` |
| Medical history | `Alcohol`, `Endemic`, `Smoking`, `Diabetes`, `Obesity`, `Hemochromatosis`, `AHT`, `CRI`, `HIV`, `Grams_day`, `Packs_year` |
| Liver disease information | `HBsAg`, `HBeAg`, `HBcAb`, `HCVAb`, `Cirrhosis`, `NASH` |
| Tumor characteristics | `PVT`, `Metastasis`, `Hallmark`, `Nodules`, `Major_Dim` |
| Laboratory values | `INR`, `AFP`, `Hemoglobin`, `MCV`, `Leucocytes`, `Platelets`, `Albumin`, `Total_Bil`, `ALT`, `AST`, `GGT`, `ALP`, `TP`, `Creatinine`, `Dir_Bil`, `Iron`, `Sat`, `Ferritin` |

Binary variables use the source dataset's documented `0`/`1` coding. The ordinal variables
`PS`, `Encephalopathy`, and `Ascites` retain their documented ordered codes. The separate
target is `Class`, where `1 = survived at one year` and `0 = died within one year`.

Only documented missing-value markers are accepted. Other nonnumeric values and fractional or
out-of-category codes are rejected rather than silently transformed. Broadly implausible values
are retained and summarized as aggregate quality flags; they are not deleted, clipped, or
corrected.

The schema is a technical data dictionary, not a clinical reference. Its broad bounds are
quality screens and must not be read as diagnostic reference intervals.
