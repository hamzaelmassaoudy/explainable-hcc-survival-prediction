"""Checked-in authoritative HCC feature schema.

Semantics and ordering follow the official UCI HCC Survival description. Binary
features use the dataset's documented 0/1 representation. Ordinal orders are used
only for variables for which the official description establishes clinical order.
"""

from dataclasses import asdict, dataclass
from typing import Literal

FeatureKind = Literal["numerical", "categorical", "ordinal"]


@dataclass(frozen=True)
class FeatureSpec:
    """Meaning and modeling treatment of one input feature."""

    name: str
    label: str
    kind: FeatureKind
    group: str
    unit: str | None = None
    categories: tuple[int, ...] | None = None
    description: str = ""


def _binary(name: str, label: str, group: str) -> FeatureSpec:
    return FeatureSpec(
        name, label, "categorical", group, categories=(0, 1), description="No (0), yes (1)."
    )


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("Gender", "Sex", "categorical", "Demographics", categories=(0, 1)),
    _binary("Symptoms", "Symptoms present", "Symptoms and clinical status"),
    _binary("Alcohol", "Alcohol use", "Medical history"),
    _binary("HBsAg", "Hepatitis B surface antigen", "Liver disease information"),
    _binary("HBeAg", "Hepatitis B e antigen", "Liver disease information"),
    _binary("HBcAb", "Hepatitis B core antibody", "Liver disease information"),
    _binary("HCVAb", "Hepatitis C virus antibody", "Liver disease information"),
    _binary("Cirrhosis", "Cirrhosis", "Liver disease information"),
    _binary("Endemic", "Residence in endemic country", "Medical history"),
    _binary("Smoking", "Smoking", "Medical history"),
    _binary("Diabetes", "Diabetes", "Medical history"),
    _binary("Obesity", "Obesity", "Medical history"),
    _binary("Hemochromatosis", "Hemochromatosis", "Medical history"),
    _binary("AHT", "Arterial hypertension", "Medical history"),
    _binary("CRI", "Chronic renal insufficiency", "Medical history"),
    _binary("HIV", "Human immunodeficiency virus", "Medical history"),
    _binary("NASH", "Nonalcoholic steatohepatitis", "Liver disease information"),
    _binary("Varices", "Esophageal varices", "Liver disease information"),
    _binary("Splenomegaly", "Splenomegaly", "Symptoms and clinical status"),
    _binary("PHT", "Portal hypertension", "Liver disease information"),
    _binary("PVT", "Portal vein thrombosis", "Tumor characteristics"),
    _binary("Metastasis", "Liver metastasis", "Tumor characteristics"),
    _binary("Hallmark", "Radiological hallmark", "Tumor characteristics"),
    FeatureSpec("Age", "Age at diagnosis", "numerical", "Demographics", "years"),
    FeatureSpec("Grams_day", "Alcohol consumption", "numerical", "Medical history", "g/day"),
    FeatureSpec("Packs_year", "Smoking exposure", "numerical", "Medical history", "pack-years"),
    FeatureSpec(
        "PS",
        "Performance status",
        "ordinal",
        "Symptoms and clinical status",
        categories=(0, 1, 2, 3, 4),
    ),
    FeatureSpec(
        "Encephalopathy",
        "Encephalopathy degree",
        "ordinal",
        "Symptoms and clinical status",
        categories=(0, 1, 2, 3),
    ),
    FeatureSpec(
        "Ascites",
        "Ascites degree",
        "ordinal",
        "Symptoms and clinical status",
        categories=(0, 1, 2, 3),
    ),
    FeatureSpec("INR", "International normalized ratio", "numerical", "Laboratory values"),
    FeatureSpec("AFP", "Alpha-fetoprotein", "numerical", "Laboratory values", "ng/mL"),
    FeatureSpec("Hemoglobin", "Hemoglobin", "numerical", "Laboratory values", "g/dL"),
    FeatureSpec("MCV", "Mean corpuscular volume", "numerical", "Laboratory values", "fL"),
    FeatureSpec("Leucocytes", "Leukocytes", "numerical", "Laboratory values", "G/L"),
    FeatureSpec("Platelets", "Platelets", "numerical", "Laboratory values", "G/L"),
    FeatureSpec("Albumin", "Albumin", "numerical", "Laboratory values", "mg/dL"),
    FeatureSpec("Total_Bil", "Total bilirubin", "numerical", "Laboratory values", "mg/dL"),
    FeatureSpec("ALT", "Alanine transaminase", "numerical", "Laboratory values", "U/L"),
    FeatureSpec("AST", "Aspartate transaminase", "numerical", "Laboratory values", "U/L"),
    FeatureSpec("GGT", "Gamma-glutamyl transferase", "numerical", "Laboratory values", "U/L"),
    FeatureSpec("ALP", "Alkaline phosphatase", "numerical", "Laboratory values", "U/L"),
    FeatureSpec("TP", "Total proteins", "numerical", "Laboratory values", "g/dL"),
    FeatureSpec("Creatinine", "Creatinine", "numerical", "Laboratory values", "mg/dL"),
    FeatureSpec("Nodules", "Number of nodules", "numerical", "Tumor characteristics", "count"),
    FeatureSpec(
        "Major_Dim", "Major dimension of nodule", "numerical", "Tumor characteristics", "cm"
    ),
    FeatureSpec("Dir_Bil", "Direct bilirubin", "numerical", "Laboratory values", "mg/dL"),
    FeatureSpec("Iron", "Iron", "numerical", "Laboratory values", "mcg/dL"),
    FeatureSpec("Sat", "Oxygen saturation", "numerical", "Laboratory values", "%"),
    FeatureSpec("Ferritin", "Ferritin", "numerical", "Laboratory values", "ng/mL"),
)

FEATURE_NAMES = tuple(spec.name for spec in FEATURE_SPECS)
NUMERICAL_FEATURES = tuple(spec.name for spec in FEATURE_SPECS if spec.kind == "numerical")
CATEGORICAL_FEATURES = tuple(spec.name for spec in FEATURE_SPECS if spec.kind == "categorical")
ORDINAL_FEATURES = tuple(spec.name for spec in FEATURE_SPECS if spec.kind == "ordinal")
ORDINAL_CATEGORIES = tuple(
    list(spec.categories or ()) for spec in FEATURE_SPECS if spec.kind == "ordinal"
)

# Predeclared before benchmark execution. This is not claimed to reproduce an established
# clinical score; it is an intentionally small, understandable comparison feature set.
REDUCED_CLINICAL_FEATURES = (
    "Age",
    "PS",
    "Ascites",
    "INR",
    "Albumin",
    "Total_Bil",
    "Creatinine",
    "AFP",
    "Nodules",
    "Major_Dim",
    "PVT",
)

# Broad plausibility flags only. Values outside these bounds are reported, never modified.
PLAUSIBILITY_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "Age": (0, 120),
    "Grams_day": (0, None),
    "Packs_year": (0, None),
    "INR": (0, None),
    "Hemoglobin": (0, 30),
    "Platelets": (0, None),
    "Albumin": (0, None),
    "Total_Bil": (0, None),
    "Creatinine": (0, None),
    "Nodules": (0, None),
    "Major_Dim": (0, None),
    "Dir_Bil": (0, None),
    "Sat": (0, 100),
}


def schema_records() -> list[dict[str, object]]:
    """Return serializable schema rows for documentation and artifacts."""

    return [asdict(spec) for spec in FEATURE_SPECS]
