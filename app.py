from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Visual design ----------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --ink: #080808;
            --paper: #f3f2ee;
            --muted: #686761;
            --line: #d7d5ce;
            --accent: #ff4b1f;
            --card: #faf9f6;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: var(--paper);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            display: none;
        }

        .block-container {
            max-width: 1380px;
            padding: 1.1rem 4.5rem 4rem;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--ink);
            padding: .55rem 0 .9rem;
            margin-bottom: 2.4rem;
        }

        .brand {
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .brand-mark {
            display: inline-block;
            width: 8px;
            height: 8px;
            margin-left: 5px;
            background: var(--accent);
        }

        .nav {
            display: flex;
            gap: 2rem;
            font-family: 'DM Mono', monospace;
            font-size: .62rem;
            text-transform: uppercase;
            letter-spacing: .08em;
        }

        .hero-wrap {
            position: relative;
            min-height: 330px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            margin-bottom: 3.4rem;
        }

        .eyebrow {
            font-family: 'DM Mono', monospace;
            font-size: .68rem;
            letter-spacing: .12em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .hero-title {
            font-size: clamp(4.5rem, 10vw, 9.2rem);
            line-height: .78;
            letter-spacing: -.075em;
            font-weight: 800;
            margin: 1.1rem 0 1.8rem;
            text-transform: uppercase;
        }

        .hero-title .accent {
            color: var(--accent);
        }

        .hero-copy {
            max-width: 610px;
            font-size: .95rem;
            line-height: 1.55;
            color: #34332f;
        }

        .hero-meta {
            position: absolute;
            right: 0;
            bottom: 0;
            width: 210px;
            font-family: 'DM Mono', monospace;
            font-size: .62rem;
            line-height: 1.7;
            text-transform: uppercase;
            color: var(--muted);
        }

        .section-rule {
            border-top: 1px solid var(--ink);
            margin: 1rem 0 1.6rem;
        }

        .section-label {
            font-family: 'DM Mono', monospace;
            font-size: .67rem;
            letter-spacing: .1em;
            text-transform: uppercase;
            margin-bottom: .65rem;
        }

        .section-heading {
            font-size: 2rem;
            line-height: 1;
            letter-spacing: -.045em;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--card);
            border: 1px solid var(--ink) !important;
            border-radius: 0 !important;
            box-shadow: 8px 8px 0 var(--ink);
            padding: .7rem .9rem;
        }

        label, .stSelectbox label, .stNumberInput label {
            font-size: .76rem !important;
            font-weight: 600 !important;
            color: var(--ink) !important;
        }

        input, [data-baseweb="select"] > div {
            border-radius: 0 !important;
            border: 1px solid #9b9991 !important;
            background: #fff !important;
            color: var(--ink) !important;
            min-height: 2.7rem;
        }

        input:focus, [data-baseweb="select"] > div:focus-within {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px var(--accent) !important;
        }

        .stNumberInput button {
            border-radius: 0 !important;
            color: var(--ink) !important;
        }

        .stButton > button {
            border-radius: 0 !important;
            border: 1px solid var(--ink) !important;
            background: var(--accent) !important;
            color: #fff !important;
            min-height: 3.2rem;
            font-size: .76rem !important;
            font-weight: 800 !important;
            letter-spacing: .08em;
            text-transform: uppercase;
            box-shadow: 5px 5px 0 var(--ink);
            transition: transform .12s ease, box-shadow .12s ease;
        }

        .stButton > button:hover {
            transform: translate(3px, 3px);
            box-shadow: 2px 2px 0 var(--ink);
        }

        .input-note {
            font-family: 'DM Mono', monospace;
            color: var(--muted);
            font-size: .58rem;
            line-height: 1.5;
            margin-top: .15rem;
            margin-bottom: .7rem;
            text-transform: uppercase;
        }

        .result-shell {
            border-top: 1px solid var(--ink);
            margin-top: 4rem;
            padding-top: 1.2rem;
        }

        .result-title {
            font-family: 'DM Mono', monospace;
            font-size: .67rem;
            letter-spacing: .1em;
            text-transform: uppercase;
            margin-bottom: 1.2rem;
        }

        .result-number {
            font-size: clamp(4rem, 8vw, 7.5rem);
            line-height: .82;
            font-weight: 800;
            letter-spacing: -.08em;
        }

        .result-unit {
            font-family: 'DM Mono', monospace;
            font-size: .7rem;
            text-transform: uppercase;
            color: var(--muted);
            margin-top: .7rem;
        }

        .result-category {
            font-size: 2.3rem;
            line-height: 1;
            font-weight: 700;
            letter-spacing: -.05em;
        }

        .result-category.accent {
            color: var(--accent);
        }

        .result-copy {
            font-family: 'DM Mono', monospace;
            font-size: .65rem;
            line-height: 1.7;
            text-transform: uppercase;
            color: var(--muted);
            max-width: 350px;
        }

        .prediction-box {
            margin-top: 1.5rem;
            padding: 1rem 1.1rem;
            border: 1px solid var(--ink);
            background: #fff;
            font-family: 'DM Mono', monospace;
            font-size: .68rem;
            letter-spacing: .06em;
            text-transform: uppercase;
        }

        .prediction-box.alert {
            border-color: var(--accent);
            box-shadow: 5px 5px 0 var(--accent);
        }

        .prediction-box.safe {
            box-shadow: 5px 5px 0 var(--ink);
        }

        .footer {
            border-top: 1px solid var(--ink);
            margin-top: 4rem;
            padding-top: 1rem;
            display: flex;
            justify-content: space-between;
            gap: 2rem;
            font-family: 'DM Mono', monospace;
            font-size: .58rem;
            line-height: 1.7;
            color: var(--muted);
            text-transform: uppercase;
        }

        @media (max-width: 900px) {
            .block-container {
                padding: 1rem 1.2rem 3rem;
            }
            .nav {
                display: none;
            }
            .hero-title {
                font-size: 4.5rem;
            }
            .hero-meta {
                position: static;
                margin-top: 2rem;
            }
            .footer {
                flex-direction: column;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
    <div class="topbar">
        <div class="brand">DIABETES LAB<span class="brand-mark"></span></div>
        <div class="nav">
            <span>MODEL</span>
            <span>INPUTS</span>
            <span>ASSESSMENT</span>
        </div>
    </div>

    <div class="hero-wrap">
        <div>
            <div class="eyebrow">Machine learning / health assessment</div>
            <div class="hero-title">RISK<br><span class="accent">ASSESSMENT</span></div>
            <div class="hero-copy">
                A streamlined interface for estimating diabetes risk from the health variables used by our trained classification model.
            </div>
        </div>
        <div class="hero-meta">
            MODEL<br>
            LOGISTIC REGRESSION<br><br>
            DATASET<br>
            DIABETES.CSV<br><br>
            OUTPUT<br>
            PROBABILITY + CATEGORY
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

TARGET_SOURCE = "glyhb"
TARGET = "Outcome"
THRESHOLD = 0.45

DROP_COLUMNS = [
    "id", "glyhb", "bp.2s", "bp.2d", "hip", "waist",
    "height", "weight", "frame", "location"
]

DEFAULTS = {
    "chol": 220.0,
    "stab.glu": 110.0,
    "hdl": 45.0,
    "ratio": 4.8,
    "age": 52,
    "bp.1s": 140.0,
    "bp.1d": 88.0,
    "time.ppn": 300.0,
    "gender": "male",
}


@st.cache_resource
def train_model():
    data_path = Path(__file__).resolve().parent / "diabetes.csv"
    if not data_path.exists():
        raise FileNotFoundError("diabetes.csv was not found in the app folder.")

    raw = pd.read_csv(data_path)

    if TARGET_SOURCE not in raw.columns:
        raise ValueError(f"The dataset must contain the '{TARGET_SOURCE}' column.")

    df = raw.dropna(subset=[TARGET_SOURCE]).copy()
    df[TARGET] = (df[TARGET_SOURCE] >= 6.5).astype(int)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    num_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET)
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].dropna().empty:
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    if y.nunique() < 2:
        raise ValueError("The dataset must contain both outcome classes.")

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train_res)

    return model, scaler, X.columns.tolist()


def make_patient(values, columns):
    row = pd.DataFrame([values])
    for col in columns:
        if col not in row.columns:
            row[col] = 0
    return row[columns]


def risk_label(probability):
    if probability >= 0.70:
        return "High"
    if probability >= THRESHOLD:
        return "Elevated"
    return "Lower"


try:
    model, scaler, columns = train_model()
except Exception as exc:
    st.error(f"Could not load the diabetes model: {exc}")
    st.stop()

# ---------- Inputs ----------
st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">01 / Patient variables</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Enter the measurements</div>', unsafe_allow_html=True)

with st.container(border=True):
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        chol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0.0,
            value=DEFAULTS["chol"],
            step=1.0,
            help="Total cholesterol concentration in milligrams per decilitre.",
        )
        st.markdown('<div class="input-note">Unit: mg/dL</div>', unsafe_allow_html=True)

        stab_glu = st.number_input(
            "Stable glucose (mg/dL)",
            min_value=0.0,
            value=DEFAULTS["stab.glu"],
            step=1.0,
            help="Glucose measurement in milligrams per decilitre.",
        )
        st.markdown('<div class="input-note">Unit: mg/dL</div>', unsafe_allow_html=True)

        hdl = st.number_input(
            "HDL (mg/dL)",
            min_value=0.0,
            value=DEFAULTS["hdl"],
            step=1.0,
            help="High-density lipoprotein cholesterol concentration.",
        )
        st.markdown('<div class="input-note">Unit: mg/dL</div>', unsafe_allow_html=True)

    with c2:
        ratio = st.number_input(
            "Cholesterol / HDL ratio",
            min_value=0.0,
            value=DEFAULTS["ratio"],
            step=0.1,
            help="Ratio of total cholesterol to HDL. This is dimensionless.",
        )
        st.markdown('<div class="input-note">Unit: ratio / dimensionless</div>', unsafe_allow_html=True)

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=DEFAULTS["age"],
            help="Age of the person being assessed.",
        )
        st.markdown('<div class="input-note">Unit: years</div>', unsafe_allow_html=True)

        gender = st.selectbox(
            "Gender",
            ["female", "male"],
            index=1,
            help="Categorical input used by the trained model.",
        )
        st.markdown('<div class="input-note">Unit: categorical</div>', unsafe_allow_html=True)

    with c3:
        bp_1s = st.number_input(
            "Systolic BP (mmHg)",
            min_value=0.0,
            value=DEFAULTS["bp.1s"],
            step=1.0,
            help="Systolic blood pressure in millimetres of mercury.",
        )
        st.markdown('<div class="input-note">Unit: mmHg</div>', unsafe_allow_html=True)

        bp_1d = st.number_input(
            "Diastolic BP (mmHg)",
            min_value=0.0,
            value=DEFAULTS["bp.1d"],
            step=1.0,
            help="Diastolic blood pressure in millimetres of mercury.",
        )
        st.markdown('<div class="input-note">Unit: mmHg</div>', unsafe_allow_html=True)

        time_ppn = st.number_input(
            "Time since previous meal (min)",
            min_value=0.0,
            value=DEFAULTS["time.ppn"],
            step=1.0,
            help="Time elapsed since the person's previous meal.",
        )
        st.markdown('<div class="input-note">Unit: minutes</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)
    assess = st.button("Run assessment", type="primary", use_container_width=True)

# ---------- Prediction ----------
if assess:
    patient_values = {
        "chol": chol,
        "stab.glu": stab_glu,
        "hdl": hdl,
        "ratio": ratio,
        "age": age,
        "bp.1s": bp_1s,
        "bp.1d": bp_1d,
        "time.ppn": time_ppn,
        "gender_male": 1 if gender == "male" else 0,
    }

    try:
        patient_df = make_patient(patient_values, columns)
        scaled_patient = scaler.transform(patient_df)
        probability = float(model.predict_proba(scaled_patient)[0][1])
        prediction = int(probability >= THRESHOLD)
        category = risk_label(probability)

        st.markdown(
            '<div class="result-shell"><div class="result-title">02 / Assessment output</div></div>',
            unsafe_allow_html=True,
        )

        r1, r2 = st.columns([1.15, 1], gap="large")

        with r1:
            st.markdown(
                f'<div class="result-number">{probability * 100:.1f}%</div>'
                '<div class="result-unit">Estimated model probability</div>',
                unsafe_allow_html=True,
            )
            st.progress(min(probability, 1.0))

        with r2:
            category_class = "accent" if category in ["Elevated", "High"] else ""
            st.markdown(
                f'<div class="result-category {category_class}">{category}</div>'
                '<div class="result-unit">Risk category</div>'
                '<div class="result-copy" style="margin-top:1.2rem;">'
                '45% is the binary classification cutoff. 70% marks the high-risk display category.'
                '</div>',
                unsafe_allow_html=True,
            )

        if prediction == 1:
            st.markdown(
                '<div class="prediction-box alert">Model prediction: DIABETIC (1)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="prediction-box safe">Model prediction: NON-DIABETIC (0)</div>',
                unsafe_allow_html=True,
            )

    except Exception as exc:
        st.error(f"Prediction error: {exc}")

# ---------- Footer ----------
st.markdown(
    """
    <div class="footer">
        <div>DIABETES LAB / MACHINE LEARNING ASSESSMENT</div>
        <div>This tool provides a model-based prediction from the supplied dataset. It is not a medical diagnosis.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
