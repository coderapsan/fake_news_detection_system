import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import joblib
import streamlit as st

from text_utils import clean_text


st.set_page_config(
    page_title="Fake News Detection Prototype",
    page_icon="📰",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
SAMPLE_MODEL_PATH = ARTIFACTS_DIR / "best_model_sample.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
SAMPLE_METADATA_PATH = ARTIFACTS_DIR / "model_metadata_sample.json"


def load_json_if_exists(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_model_bundle():
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            metadata = load_json_if_exists(METADATA_PATH)
            source = "full training run"
        except ValueError as e:
            if "unsupported pickle protocol" in str(e):
                st.warning("Model file is incompatible with Python 3.7. Please retrain the model using train_pipeline.py.")
                st.stop()
            raise
    elif SAMPLE_MODEL_PATH.exists():
        model = joblib.load(SAMPLE_MODEL_PATH)
        metadata = load_json_if_exists(SAMPLE_METADATA_PATH)
        source = "included sample model"
    else:
        st.error("No trained model found. Run train_pipeline.py first.")
        st.stop()

    return model, metadata, source


def pseudo_confidence(model, text: str) -> Optional[float]:
    clf = model.named_steps.get("clf")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        return float(max(proba))

    if hasattr(clf, "decision_function"):
        transformed = model.named_steps["tfidf"].transform([text])
        score = float(clf.decision_function(transformed)[0])
        return 1.0 / (1.0 + math.exp(-abs(score)))

    return None


def get_word_count(text: str) -> int:
    return len(str(text).split())


def get_char_count(text: str) -> int:
    return len(str(text))


def get_input_quality_message(title: str, body: str) -> Tuple[str, str]:
    combined = f"{title} {body}".strip()
    wc = get_word_count(combined)
    cc = get_char_count(combined)

    if not combined:
        return "warning", "Please enter some text first."

    if wc < 20 or cc < 100:
        return "warning", "Input is very short. The prediction may be less reliable."

    if wc < 50:
        return "info", "Input is moderate in length. Adding more article body text may improve reliability."

    return "success", "Input length looks good for prediction."


def extract_present_keywords(model, text: str, top_n: int = 12):
    try:
        tfidf = model.named_steps.get("tfidf")
        clf = model.named_steps.get("clf")

        if tfidf is None or clf is None:
            return []

        if not hasattr(tfidf, "get_feature_names_out"):
            return []

        if not hasattr(clf, "coef_"):
            return []

        cleaned = clean_text(text)
        X = tfidf.transform([cleaned])
        feature_names = tfidf.get_feature_names_out()
        coef = clf.coef_[0]

        row = X.tocoo()
        present = []
        for col_idx, tfidf_value in zip(row.col, row.data):
            term = feature_names[col_idx]
            weight = coef[col_idx] * tfidf_value
            present.append((term, float(weight)))

        # Larger positive weights generally push toward class 1 (Fake)
        present = sorted(present, key=lambda x: abs(x[1]), reverse=True)
        return present[:top_n]

    except Exception:
        return []


def build_result_payload(title: str, body: str, label: str, confidence: Optional[float], source: str):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "title": title,
        "body_preview": body[:300],
        "prediction": label,
        "confidence": None if confidence is None else round(confidence, 4),
        "model_source": source,
    }


def inject_custom_css():
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.7rem;
                font-weight: 800;
                margin-bottom: 0.2rem;
            }
            .subtitle {
                color: #6b7280;
                font-size: 1rem;
                margin-bottom: 1.2rem;
            }
            .card {
                padding: 1rem 1.2rem;
                border-radius: 16px;
                background: #f8fafc;
                border: 1px solid #e5e7eb;
                margin-bottom: 0.8rem;
            }
            .metric-label {
                font-size: 0.9rem;
                color: #6b7280;
                margin-bottom: 0.2rem;
            }
            .metric-value {
                font-size: 1.8rem;
                font-weight: 700;
            }
            .fake-box {
                background: #fef2f2;
                border: 1px solid #fecaca;
                color: #991b1b;
                border-radius: 14px;
                padding: 1rem;
                font-weight: 700;
            }
            .real-box {
                background: #ecfdf5;
                border: 1px solid #a7f3d0;
                color: #065f46;
                border-radius: 14px;
                padding: 1rem;
                font-weight: 700;
            }
            .small-note {
                color: #6b7280;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


model, metadata, source = load_model_bundle()
inject_custom_css()

if "title_text" not in st.session_state:
    st.session_state.title_text = ""
if "body_text" not in st.session_state:
    st.session_state.body_text = ""


def load_fake_example():
    st.session_state.title_text = "Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing"
    st.session_state.body_text = (
        "Donald Trump just couldn’t wish all Americans a Happy New Year and leave it at that. "
        "Instead, he had to include a political message that turns the article into a partisan and sensational claim."
    )


def load_real_example():
    st.session_state.title_text = "Central bank releases updated quarterly inflation report"
    st.session_state.body_text = (
        "The central bank on Thursday published its quarterly inflation report, outlining current price trends, "
        "monetary policy decisions, and updated projections for the coming year."
    )


st.markdown('<div class="main-title">📰 Fake News Detection Prototype</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Machine learning prototype for automated article classification with an improved interface and analysis panel.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model Panel")
    st.write(f"**Loaded from:** {source}")

    if metadata:
        with st.expander("View model metadata", expanded=False):
            st.json(metadata)

    st.markdown("---")
    st.subheader("Quick Demo")
    if st.button("Load fake-style example"):
        load_fake_example()
    if st.button("Load real-style example"):
        load_real_example()

    st.markdown("---")
    st.subheader("About")
    st.caption(
        "This tool supports article screening. It should assist human review, not replace fact-checking."
    )

tab1, tab2, tab3 = st.tabs(["Detector", "Analysis", "About the Model"])

with tab1:
    col_left, col_right = st.columns([2.2, 1])

    with col_left:
        title = st.text_input(
            "Article title",
            key="title_text",
            placeholder="Paste the headline here",
        )

        body = st.text_area(
            "Article body",
            key="body_text",
            height=260,
            placeholder="Paste the article text here",
        )

        quality_type, quality_msg = get_input_quality_message(title, body)
        if quality_type == "warning":
            st.warning(quality_msg)
        elif quality_type == "info":
            st.info(quality_msg)
        else:
            st.success(quality_msg)

        predict = st.button("Predict", use_container_width=True)

    with col_right:
        combined_preview = f"{title} {body}".strip()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Word count</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{get_word_count(combined_preview)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Character count</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{get_char_count(combined_preview)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Preprocessed preview</div>', unsafe_allow_html=True)
        st.caption(clean_text(combined_preview)[:250] if combined_preview else "Nothing entered yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    if predict:
        combined = f"{title} {body}".strip()

        if not combined:
            st.warning("Please enter a title, body, or both.")
        else:
            pred = int(model.predict([combined])[0])
            label = "Fake" if pred == 1 else "Real"
            confidence = pseudo_confidence(model, combined)

            st.markdown("### Prediction Result")
            if pred == 1:
                st.markdown(f'<div class="fake-box">Prediction: {label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="real-box">Prediction: {label}</div>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Prediction", label)

            with m2:
                if confidence is not None:
                    st.metric("Confidence", f"{confidence:.2%}")
                else:
                    st.metric("Confidence", "N/A")

            with m3:
                st.metric("Words analysed", get_word_count(combined))

            if confidence is not None:
                st.progress(min(max(float(confidence), 0.0), 1.0))
                st.caption("Confidence bar")

            result_payload = build_result_payload(title, body, label, confidence, source)

            st.download_button(
                "Download result as JSON",
                data=json.dumps(result_payload, indent=2),
                file_name="prediction_result.json",
                mime="application/json",
            )

            st.info(
                "This output is probabilistic. It should support human fact-checking rather than replace it."
            )

with tab2:
    st.subheader("Article Analysis")

    combined = f"{st.session_state.title_text} {st.session_state.body_text}".strip()

    if combined:
        keywords = extract_present_keywords(model, combined, top_n=12)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Cleaned Text Preview")
            st.code(clean_text(combined)[:1000] or "No processed text available")

        with col2:
            st.markdown("#### Indicative Terms Found")
            if keywords:
                for term, weight in keywords:
                    direction = "Fake-side" if weight > 0 else "Real-side"
                    st.write(f"**{term}** → {direction} ({weight:.4f})")
            else:
                st.caption("Keyword explanation is not available for the current trained model type.")

        st.markdown("#### Input Statistics")
        stats = {
            "Title words": get_word_count(st.session_state.title_text),
            "Body words": get_word_count(st.session_state.body_text),
            "Total words": get_word_count(combined),
            "Total characters": get_char_count(combined),
        }
        st.json(stats)
    else:
        st.info("Enter some article text in the Detector tab to see analysis.")

with tab3:
    st.subheader("About this App")
    st.write(
        """
        This interface uses a trained text-classification pipeline to label an article as likely **Fake** or **Real**.
        The app combines headline and body text, preprocesses the input, transforms it with TF-IDF features,
        and then uses the saved trained model for classification.
        """
    )