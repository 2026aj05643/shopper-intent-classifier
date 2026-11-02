"""Streamlit app for ML Assignment 2 (Saheb Chatterjee, 2026AJ05643).

Upload the test CSV, pick one of the five saved classifiers and see how it does.
Nothing is trained here: the models in model/ were fitted in model_training.ipynb.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
SAMPLE_CSV = HERE / "test_data.csv"

st.set_page_config(page_title="Shopper intent - model check", layout="wide")


@st.cache_resource
def load_models():
    info = json.loads((MODEL_DIR / "model_info.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname)
              for name, fname in info["models"].items()}
    return info, models


@st.cache_data
def read_csv(source):
    return pd.read_csv(source)


def to_binary(col):
    """Revenue may come in as True/False, TRUE/FALSE or 1/0."""
    mapped = col.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    return mapped


def scores(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob) if y.nunique() == 2 else np.nan,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


info, models = load_models()
features, target = info["features"], info["target"]

# ---------------- sidebar ----------------
st.sidebar.header("Test data")
upload = st.sidebar.file_uploader("Upload a test CSV", type=["csv"])
st.sidebar.download_button(
    "Download my test_data.csv", SAMPLE_CSV.read_bytes(),
    file_name="test_data.csv", mime="text/csv",
    help="The 20% hold-out split from the notebook. Upload it back to try the app.")
use_sample = st.sidebar.checkbox(
    "No file? Use the bundled test_data.csv", value=upload is None,
    disabled=upload is not None)

st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Classifier", list(models))
threshold = st.sidebar.slider(
    "Call it a purchase when P(purchase) is at least", 0.05, 0.95, 0.50, 0.05,
    help="0.50 is what the README table uses. Lower it to catch more buyers, "
         "raise it to make fewer false alarms.")

# ---------------- header ----------------
st.title("Will this visitor buy?")
st.write(
    "Five classifiers trained on the UCI *Online Shoppers Purchasing Intention* data "
    "predict whether a browsing session ends in a purchase (`Revenue`). Upload test "
    "data with the same 17 columns plus `Revenue`; the saved models are only used "
    "to predict, never retrained.")

if upload is not None:
    data, source_name = pd.read_csv(upload), upload.name
elif use_sample:
    data, source_name = read_csv(SAMPLE_CSV), "test_data.csv (bundled)"
else:
    st.info("Upload a CSV in the sidebar, or tick the box to use the bundled test data.")
    st.stop()

missing = [c for c in features + [target] if c not in data.columns]
if missing:
    st.error("These columns are missing from the file: " + ", ".join(missing))
    st.stop()

y = to_binary(data[target])
if y.isna().any():
    st.error(f"`{target}` should only hold True/False or 1/0 values. "
             f"{int(y.isna().sum())} rows have something else.")
    st.stop()
y = y.astype(int)
X = data[features]

c1, c2, c3 = st.columns(3)
c1.metric("Rows in file", f"{len(data):,}")
c2.metric("Sessions with a purchase", f"{int(y.sum()):,}")
c3.metric("Purchase rate", f"{y.mean():.1%}")
st.caption(f"Source: {source_name}")

tab_one, tab_all, tab_data = st.tabs(
    ["Selected model", "All five models", "Data preview"])

# ---------------- one model ----------------
with tab_one:
    model = models[model_name]
    prob = model.predict_proba(X)[:, 1]
    pred, m = scores(y, prob, threshold)

    st.subheader(f"{model_name}  (threshold {threshold:.2f})")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No purchase", "Purchase"],
                    yticklabels=["No purchase", "Purchase"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"{tp} buyers caught, {fn} buyers missed, "
                   f"{fp} non-buyers flagged by mistake.")
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y, pred, labels=[0, 1],
                                    target_names=["No purchase", "Purchase"],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y, prob)
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {m['AUC']:.3f}")
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    out = data.copy()
    out["P(purchase)"] = prob.round(4)
    out["Predicted"] = np.where(pred == 1, "Purchase", "No purchase")
    st.download_button("Download predictions as CSV",
                       out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.replace(' ', '_')}.csv",
                       mime="text/csv")

# ---------------- all models ----------------
with tab_all:
    st.subheader(f"All five models on this file (threshold {threshold:.2f})")
    probs = {name: mdl.predict_proba(X)[:, 1] for name, mdl in models.items()}
    table = pd.DataFrame({name: scores(y, p, threshold)[1]
                          for name, p in probs.items()}).T
    table.index.name = "Model"
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8d5"),
                 use_container_width=True)
    st.caption("Green marks the best model for each metric.")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        table[["Accuracy", "F1", "MCC"]].plot.bar(ax=ax, rot=20, width=0.75)
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.legend(loc="lower right", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for name, p in probs.items():
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, lw=1.5, label=f"{name} ({table.loc[name, 'AUC']:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7, loc="lower right")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

# ---------------- data ----------------
with tab_data:
    st.write(f"{len(data):,} rows, {data.shape[1]} columns. First 50 rows:")
    st.dataframe(data.head(50), use_container_width=True)
    extra = [c for c in data.columns if c not in features + [target]]
    if extra:
        st.caption("Ignored extra columns: " + ", ".join(extra))
