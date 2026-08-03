"""Streamlit research demonstration for the fitted HCC model."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from hcc_survival.constants import DEFAULT_MODEL_PATH
from hcc_survival.prediction import (
    MODEL_RECOVERY_COMMAND,
    ModelArtifactError,
    PredictionInputError,
    load_model_bundle,
    predict_survival,
)
from hcc_survival.schemas import FEATURE_SPECS

st.set_page_config(page_title="HCC Survival Research Demo", page_icon="🔬", layout="wide")
st.title("Explainable HCC one-year survival research demo")
st.error(
    "Research and education only. This model is not clinically validated and must not "
    "be used for diagnosis, treatment, triage, or medical decisions."
)
st.write(
    "This prototype estimates the probability of surviving one year (`1 = survived`). "
    "It does not provide medical advice, and missing values are allowed."
)

try:
    bundle = load_model_bundle(DEFAULT_MODEL_PATH)
except (FileNotFoundError, ModelArtifactError) as exc:
    st.info(str(exc))
    st.caption(
        "Provisional local model recovery only: this command creates a research artifact, "
        "not a clinically validated model."
    )
    st.code(
        "python -m hcc_survival download\n" + MODEL_RECOVERY_COMMAND,
        language="bash",
    )
    st.stop()

single_tab, batch_tab = st.tabs(["Single patient", "Batch CSV"])

with single_tab:
    st.subheader("Single-patient input")
    values: dict[str, float] = {}
    grouped: dict[str, list[object]] = {}
    for spec in FEATURE_SPECS:
        grouped.setdefault(spec.group, []).append(spec)
    for group_name, specs in grouped.items():
        with st.expander(group_name, expanded=group_name == "Demographics"):
            columns = st.columns(2)
            for index, spec in enumerate(specs):
                with columns[index % 2]:
                    unknown = st.checkbox("Unknown", value=True, key=f"{spec.name}_unknown")
                    if unknown:
                        values[spec.name] = np.nan
                    elif spec.kind in {"categorical", "ordinal"}:
                        values[spec.name] = float(
                            st.selectbox(
                                spec.label,
                                options=list(spec.categories or ()),
                                key=spec.name,
                                help=spec.description,
                            )
                        )
                    else:
                        values[spec.name] = float(
                            st.number_input(
                                f"{spec.label}{f' ({spec.unit})' if spec.unit else ''}",
                                value=0.0,
                                key=spec.name,
                            )
                        )
    if st.button("Estimate one-year survival probability", type="primary"):
        try:
            result = predict_survival(bundle, pd.DataFrame([values]))
            for warning in result.attrs.get("plausibility_warnings", ()):
                st.warning(warning)
            survival = result.iloc[0]["model_estimated_one_year_survival_probability"]
            st.metric("Model-estimated one-year survival probability", f"{survival:.1%}")
            st.caption(
                f"Secondary display: model-estimated one-year mortality probability = "
                f"{1 - survival:.1%}. This output is uncertain and is not medical advice."
            )
            st.info(
                "Influential-input explanations describe model behavior, not causes. Run the "
                "`explain` CLI command for validation-aligned permutation importance."
            )
        except PredictionInputError as exc:
            st.warning(str(exc))
        except Exception:
            st.error(
                "A local prediction error occurred and no result was produced. Regenerate the "
                "provisional local model before trying again."
            )

with batch_tab:
    st.subheader("Batch CSV prediction")
    st.warning(
        "Use only synthetic or non-identifiable data. This local demonstration is not a "
        "secure health-data service."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            input_frame = pd.read_csv(uploaded)
            output_frame = predict_survival(bundle, input_frame)
            for warning in output_frame.attrs.get("plausibility_warnings", ()):
                st.warning(warning)
            st.dataframe(output_frame)
            buffer = io.StringIO()
            output_frame.to_csv(buffer, index=False)
            st.download_button(
                "Download batch results",
                data=buffer.getvalue(),
                file_name="hcc_research_predictions.csv",
                mime="text/csv",
            )
        except PredictionInputError as exc:
            st.warning(f"Could not process the CSV: {exc}")
        except (pd.errors.ParserError, UnicodeError, ValueError):
            st.warning("Could not read the CSV. Save it as a standard UTF-8 CSV and try again.")
        except Exception:
            st.error(
                "A local processing error occurred and no prediction file was created. "
                "Regenerate the provisional local model before trying again."
            )

st.divider()
st.caption(
    "Do not upload real or identifiable health information. This research demonstration must "
    "not be deployed for clinical use."
)
