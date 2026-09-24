import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CreditIQ | Credit Approval Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HTML RENDERER
# IMPORTANT:
# Uses st.html() instead of st.markdown() so HTML is rendered
# as HTML and NEVER appears as literal <div> text.
# ============================================================

def render_html(html_content):
    st.html(html_content)


# ============================================================
# CUSTOM CSS
# ============================================================

render_html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500;600&display=swap');

.stApp {
    background:
        radial-gradient(
            circle at 5% 0%,
            rgba(59, 130, 246, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 95% 5%,
            rgba(34, 211, 238, 0.08),
            transparent 28%
        ),
        #07111f;
    color: #e5e7eb;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #091525 0%,
            #0b1728 50%,
            #08111e 100%
        );
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f8fafc;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #94a3b8;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(148, 163, 184, 0.12);
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;
    overflow: hidden;

    padding: 2.7rem;
    margin-bottom: 1.8rem;

    background:
        linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.98),
            rgba(15, 35, 65, 0.96)
        );

    border: 1px solid rgba(96, 165, 250, 0.22);
    border-radius: 24px;

    box-shadow:
        0 20px 60px rgba(0, 0, 0, 0.30),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.hero-badge {
    display: inline-block;

    padding: 0.4rem 0.85rem;
    margin-bottom: 1rem;

    background: rgba(34, 211, 238, 0.09);
    border: 1px solid rgba(34, 211, 238, 0.22);
    border-radius: 999px;

    color: #67e8f9;

    font-family: 'Roboto Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
}

.hero-title {
    margin: 0;

    font-family: 'Inter', sans-serif;
    font-size: clamp(2rem, 4vw, 3.4rem);
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -1.8px;

    color: #f8fafc;
}

.hero-subtitle {
    max-width: 820px;

    margin-top: 1rem;

    color: #94a3b8;

    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    line-height: 1.7;
}

.hero-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    margin-top: 1.3rem;
    padding: 0.55rem 0.9rem;

    border-radius: 10px;

    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.18);

    color: #86efac;

    font-family: 'Roboto Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
}

.status-dot {
    width: 8px;
    height: 8px;

    display: inline-block;

    border-radius: 50%;

    background: #22c55e;

    box-shadow:
        0 0 12px rgba(34, 197, 94, 0.8);
}


/* ============================================================
   SECTION HEADINGS
   ============================================================ */

.section-heading {
    margin-top: 2rem;
    margin-bottom: 0.3rem;

    color: #f8fafc;

    font-family: 'Inter', sans-serif;
    font-size: 1.45rem;
    font-weight: 750;
}

.section-description {
    margin-bottom: 1.2rem;

    color: #64748b;

    font-size: 0.86rem;
}


/* ============================================================
   KPI CARDS
   ============================================================ */

.kpi-card {
    min-height: 145px;

    padding: 1.25rem;

    background:
        linear-gradient(
            145deg,
            rgba(15, 30, 50, 0.96),
            rgba(10, 22, 37, 0.96)
        );

    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 18px;

    box-shadow:
        0 12px 30px rgba(0, 0, 0, 0.18),
        inset 0 1px 0 rgba(255, 255, 255, 0.025);
}

.kpi-label {
    color: #64748b;

    font-family: 'Roboto Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.8px;
}

.kpi-value {
    margin-top: 0.65rem;

    color: #f8fafc;

    font-size: 1.85rem;
    font-weight: 800;
}

.kpi-subtitle {
    margin-top: 0.45rem;

    color: #64748b;

    font-size: 0.72rem;
}

.kpi-blue {
    border-top: 3px solid #3b82f6;
}

.kpi-cyan {
    border-top: 3px solid #22d3ee;
}

.kpi-green {
    border-top: 3px solid #22c55e;
}

.kpi-red {
    border-top: 3px solid #f43f5e;
}

.kpi-purple {
    border-top: 3px solid #a78bfa;
}


/* ============================================================
   INSIGHT CARDS
   ============================================================ */

.insight-card {
    min-height: 205px;

    padding: 1.35rem;

    background:
        linear-gradient(
            145deg,
            rgba(15, 30, 50, 0.96),
            rgba(10, 22, 37, 0.96)
        );

    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 18px;

    box-shadow:
        0 12px 35px rgba(0, 0, 0, 0.14);
}

.insight-icon {
    font-size: 1.5rem;
    margin-bottom: 0.7rem;
}

.insight-title {
    color: #f8fafc;
    font-size: 1rem;
    font-weight: 700;
}

.insight-text {
    margin-top: 0.65rem;

    color: #cbd5e1;

    font-size: 0.87rem;
    line-height: 1.65;
}

.insight-caption {
    margin-top: 1rem;

    color: #64748b;

    font-size: 0.73rem;
    line-height: 1.5;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    min-height: 46px;

    border: 1px solid rgba(59, 130, 246, 0.40);
    border-radius: 12px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #0891b2
        );

    color: white;

    font-weight: 700;

    box-shadow:
        0 8px 24px rgba(37, 99, 235, 0.18);
}

.stButton > button:hover {
    border-color: #67e8f9;
}


/* ============================================================
   INPUTS
   ============================================================ */

.stSelectbox label,
.stMultiSelect label,
.stNumberInput label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

[data-baseweb="select"] > div {
    background-color: #0f1b2d;
    border-color: rgba(148, 163, 184, 0.16);
    border-radius: 10px;
}


/* ============================================================
   EXPANDERS
   ============================================================ */

[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.70);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 15px;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.12);
}


/* ============================================================
   FOOTER
   ============================================================ */

.custom-footer {
    margin-top: 2rem;
    padding: 1.2rem;

    text-align: center;

    border-top: 1px solid rgba(148, 163, 184, 0.10);

    color: #64748b;

    font-size: 0.75rem;
}

.footer-brand {
    color: #93c5fd;
    font-weight: 700;
}

</style>
""")


# ============================================================
# DATASET CONFIGURATION
# ============================================================

COLUMN_NAMES = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10",
    "A11",
    "A12",
    "A13",
    "A14",
    "Approval"
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    possible_files = [
        "crx.data",
        "credit.data",
        "credit.csv",
        "credit_approval.csv",
        "dataset.csv"
    ]

    selected_file = None

    for filename in possible_files:

        if os.path.exists(filename):
            selected_file = filename
            break

    if selected_file is None:

        st.error(
            "Dataset not found. Please keep crx.data "
            "in the same folder as this Python file."
        )

        st.stop()

    try:

        data = pd.read_csv(
            selected_file,
            header=None,
            names=COLUMN_NAMES,
            na_values=[
                "?",
                " ?",
                "",
                "NA",
                "N/A"
            ]
        )

    except Exception as error:

        st.error(
            f"Could not load the dataset: {error}"
        )

        st.stop()

    return data, selected_file


df, selected_file = load_data()


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(data):

    data = data.copy()

    for column in data.columns:

        if data[column].dtype == "object":

            data[column] = (
                data[column]
                .astype(str)
                .str.strip()
            )

    data.replace(
        [
            "?",
            "nan",
            "None",
            "NA",
            "N/A",
            ""
        ],
        np.nan,
        inplace=True
    )

    numeric_columns = [
        "A2",
        "A3",
        "A8",
        "A11",
        "A14"
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data["Approval"] = (
        data["Approval"]
        .astype(str)
        .str.strip()
        .map(
            {
                "+": "Approved",
                "-": "Rejected"
            }
        )
    )

    data = data.dropna(
        subset=["Approval"]
    )

    return data


df = clean_data(df)


# ============================================================
# DATASET VALIDATION
# ============================================================

if df.empty:

    st.error(
        "The dataset is empty after cleaning."
    )

    st.stop()


if df["Approval"].nunique() < 2:

    st.error(
        "The dataset must contain both Approved "
        "and Rejected records."
    )

    st.stop()


# ============================================================
# MODEL PREPARATION
# ============================================================

FEATURES = [
    column
    for column in df.columns
    if column != "Approval"
]

X = df[FEATURES]
y = df["Approval"]


numeric_features = (
    X.select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)


categorical_features = [
    column
    for column in FEATURES
    if column not in numeric_features
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=250,
    random_state=42,
    class_weight="balanced",
    max_depth=12,
    min_samples_split=4
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# TRAIN
# ============================================================

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = pipeline.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    pos_label="Approved",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    pos_label="Approved",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label="Approved",
    zero_division=0
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html("""
    <div style="
        padding: 0.4rem 0 1.2rem 0;
    ">

        <div style="
            font-size: 1.65rem;
            font-weight: 800;
            color: #f8fafc;
        ">
            💳 CreditIQ
        </div>

        <div style="
            color: #64748b;
            font-size: 0.75rem;
            margin-top: 4px;
            font-family: 'Roboto Mono', monospace;
        ">
            CREDIT APPROVAL INTELLIGENCE
        </div>

    </div>
    """)

    st.divider()

    st.subheader(
        "Dashboard Controls"
    )

    approval_filter = st.multiselect(
        "Approval Status",
        options=[
            "Approved",
            "Rejected"
        ],
        default=[
            "Approved",
            "Rejected"
        ]
    )

    st.divider()

    st.subheader(
        "Dataset"
    )

    st.write(
        "UCI Credit Approval Dataset"
    )

    st.metric(
        "Applications",
        f"{len(df):,}"
    )

    st.divider()

    st.subheader(
        "Model"
    )

    st.write(
        "Random Forest Classifier"
    )

    st.caption(
        "Python • Streamlit • Pandas • "
        "Scikit-learn • Plotly"
    )


# ============================================================
# FILTER
# ============================================================

filtered_df = df[
    df["Approval"].isin(
        approval_filter
    )
].copy()


# ============================================================
# HERO
# ============================================================

render_html("""
<div class="hero">

    <div class="hero-badge">
        AI-POWERED CREDIT ANALYTICS
    </div>

    <h1 class="hero-title">
        Credit Approval Intelligence
    </h1>

    <div class="hero-subtitle">
        Transform credit application data into
        actionable insights, risk signals and
        machine-learning assisted approval predictions.
    </div>

    <div class="hero-status">
        <span class="status-dot"></span>
        LIVE ANALYTICS DASHBOARD
    </div>

</div>
""")


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_applications = len(
    filtered_df
)

approved_count = int(
    (
        filtered_df["Approval"]
        == "Approved"
    ).sum()
)

rejected_count = int(
    (
        filtered_df["Approval"]
        == "Rejected"
    ).sum()
)

if total_applications > 0:

    approval_rate = (
        approved_count
        / total_applications
        * 100
    )

else:

    approval_rate = 0


missing_values = int(
    filtered_df.isna()
    .sum()
    .sum()
)

model_accuracy = accuracy * 100


# ============================================================
# KPI SECTION
# ============================================================

render_html("""
<div class="section-heading">
    📊 Key Performance Indicators
</div>

<div class="section-description">
    Real-time summary of the currently selected application data.
</div>
""")


k1, k2, k3, k4, k5 = st.columns(5)


with k1:

    render_html(f"""
    <div class="kpi-card kpi-blue">

        <div class="kpi-label">
            Applications
        </div>

        <div class="kpi-value">
            {total_applications:,}
        </div>

        <div class="kpi-subtitle">
            Filtered records
        </div>

    </div>
    """)


with k2:

    render_html(f"""
    <div class="kpi-card kpi-cyan">

        <div class="kpi-label">
            Approval Rate
        </div>

        <div class="kpi-value">
            {approval_rate:.1f}%
        </div>

        <div class="kpi-subtitle">
            Current approval ratio
        </div>

    </div>
    """)


with k3:

    render_html(f"""
    <div class="kpi-card kpi-green">

        <div class="kpi-label">
            Approved
        </div>

        <div class="kpi-value">
            {approved_count:,}
        </div>

        <div class="kpi-subtitle">
            Approved applications
        </div>

    </div>
    """)


with k4:

    render_html(f"""
    <div class="kpi-card kpi-red">

        <div class="kpi-label">
            Rejected
        </div>

        <div class="kpi-value">
            {rejected_count:,}
        </div>

        <div class="kpi-subtitle">
            Rejected applications
        </div>

    </div>
    """)


with k5:

    render_html(f"""
    <div class="kpi-card kpi-purple">

        <div class="kpi-label">
            Model Accuracy
        </div>

        <div class="kpi-value">
            {model_accuracy:.1f}%
        </div>

        <div class="kpi-subtitle">
            Held-out test set
        </div>

    </div>
    """)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

render_html("""
<div class="section-heading">
    📈 Executive Overview
</div>

<div class="section-description">
    High-level view of credit application outcomes
    and machine-learning performance.
</div>
""")


col1, col2 = st.columns(2)


# ============================================================
# APPLICATION OUTCOME
# ============================================================

with col1:

    st.subheader(
        "Application Outcome"
    )

    if not filtered_df.empty:

        outcome_counts = (
            filtered_df["Approval"]
            .value_counts()
            .reset_index()
        )

        outcome_counts.columns = [
            "Approval",
            "Count"
        ]

        fig = px.pie(
            outcome_counts,
            names="Approval",
            values="Count",
            hole=0.58,
            color="Approval",
            color_discrete_map={
                "Approved": "#22c55e",
                "Rejected": "#f43f5e"
            }
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#cbd5e1"
            ),
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            legend_title="Status"
        )

        fig.update_traces(
            textinfo="percent+label"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    else:

        st.warning(
            "No applications match the selected filter."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

with col2:

    st.subheader(
        "Model Performance"
    )

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],
            "Score": [
                accuracy * 100,
                precision * 100,
                recall * 100,
                f1 * 100
            ]
        }
    )

    fig = px.bar(
        metrics_df,
        x="Metric",
        y="Score",
        text="Score"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#cbd5e1"
        ),
        yaxis_title="Score (%)",
        xaxis_title="",
        yaxis_range=[
            0,
            110
        ],
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        ),
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.10)"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# APPLICATION PROFILE ANALYSIS
# ============================================================

render_html("""
<div class="section-heading">
    🔎 Application Profile Analysis
</div>

<div class="section-description">
    Explore numerical characteristics of the credit applications.
</div>
""")


numeric_display = (
    filtered_df
    .select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)


if numeric_display:

    selected_feature = st.selectbox(
        "Select a numerical variable",
        numeric_display
    )

    fig = px.histogram(
        filtered_df,
        x=selected_feature,
        color="Approval",
        nbins=30,
        marginal="box",
        color_discrete_map={
            "Approved": "#22c55e",
            "Rejected": "#f43f5e"
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(
            color="#cbd5e1"
        ),
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        legend_title="Outcome",
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.08)"
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.08)"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )

else:

    st.warning(
        "No numerical features are available."
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

render_html("""
<div class="section-heading">
    💡 Business Insights & Actions
</div>

<div class="section-description">
    Data-driven observations from the current dataset.
</div>
""")


insight1, insight2, insight3 = st.columns(3)


with insight1:

    render_html(f"""
    <div class="insight-card">

        <div class="insight-icon">📊</div>

        <div class="insight-title">
            Approval Pattern
        </div>

        <div class="insight-text">
            {approval_rate:.1f}% of the currently
            filtered applications are approved.
        </div>

        <div class="insight-caption">
            Monitor approval patterns across
            different applicant groups.
        </div>

    </div>
    """)


with insight2:

    render_html(f"""
    <div class="insight-card">

        <div class="insight-icon">🤖</div>

        <div class="insight-title">
            Predictive Signal
        </div>

        <div class="insight-text">
            The Random Forest model achieves
            {model_accuracy:.1f}% accuracy on
            the held-out test set.
        </div>

        <div class="insight-caption">
            Model predictions should support
            responsible credit assessment.
        </div>

    </div>
    """)


with insight3:

    render_html(f"""
    <div class="insight-card">

        <div class="insight-icon">⚠️</div>

        <div class="insight-title">
            Data Quality
        </div>

        <div class="insight-text">
            The filtered dataset contains
            {missing_values:,} missing values.
        </div>

        <div class="insight-caption">
            Missing values are automatically
            handled by the ML preprocessing pipeline.
        </div>

    </div>
    """)


# ============================================================
# MODEL VALIDATION
# ============================================================

render_html("""
<div class="section-heading">
    🧪 Model Validation
</div>

<div class="section-description">
    Confusion matrix showing classification
    results on the test dataset.
</div>
""")


cm = confusion_matrix(
    y_test,
    predictions,
    labels=[
        "Approved",
        "Rejected"
    ]
)


cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Approved",
        "Actual Rejected"
    ],
    columns=[
        "Predicted Approved",
        "Predicted Rejected"
    ]
)


fig = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale="Blues",
    aspect="auto"
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="#cbd5e1"
    ),
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# ============================================================
# AI CREDIT PREDICTION
# ============================================================

render_html("""
<div class="section-heading">
    🤖 AI-Assisted Credit Prediction
</div>

<div class="section-description">
    Enter applicant attributes to generate
    a Random Forest model prediction.
</div>
""")


with st.expander(
    "🚀 Open Prediction Tool"
):

    input_data = {}

    prediction_columns = st.columns(3)

    for index, feature in enumerate(FEATURES):

        current_column = prediction_columns[
            index % 3
        ]

        with current_column:

            if feature in numeric_features:

                available_values = (
                    df[feature]
                    .dropna()
                )

                if not available_values.empty:

                    default_value = float(
                        available_values.median()
                    )

                else:

                    default_value = 0.0

                input_data[feature] = st.number_input(
                    feature,
                    value=default_value
                )

            else:

                values = (
                    df[feature]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                values = sorted(values)

                if not values:

                    values = [
                        "Unknown"
                    ]

                input_data[feature] = st.selectbox(
                    feature,
                    values
                )

    predict_button = st.button(
        "🔍 Predict Credit Approval",
        use_container_width=True
    )

    if predict_button:

        applicant = pd.DataFrame(
            [input_data],
            columns=FEATURES
        )

        prediction = pipeline.predict(
            applicant
        )[0]

        probabilities = pipeline.predict_proba(
            applicant
        )[0]

        classes = pipeline.classes_

        probability_dict = dict(
            zip(
                classes,
                probabilities
            )
        )

        confidence = (
            probability_dict.get(
                prediction,
                max(probabilities)
            )
            * 100
        )

        if prediction == "Approved":

            st.success(
                f"### ✅ Prediction: Approved\n\n"
                f"Model confidence: **{confidence:.1f}%**"
            )

        else:

            st.error(
                f"### ❌ Prediction: Rejected\n\n"
                f"Model confidence: **{confidence:.1f}%**"
            )


# ============================================================
# DATA EXPLORER
# ============================================================

render_html("""
<div class="section-heading">
    📋 Application Data Explorer
</div>

<div class="section-description">
    Filtered records from the source dataset.
</div>
""")


if not filtered_df.empty:

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=360
    )

else:

    st.warning(
        "No records available for the selected filters."
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander(
    "📁 Dataset & Project Information"
):

    info1, info2 = st.columns(2)

    with info1:

        st.subheader(
            "Dataset"
        )

        st.write(
            "UCI Credit Approval Dataset"
        )

        st.write(
            "The dataset contains credit application "
            "records with numerical and categorical "
            "applicant attributes and a binary credit "
            "approval outcome."
        )

        st.write(
            f"Source file: `{selected_file}`"
        )

        st.write(
            f"Records: {len(df):,}"
        )

        st.write(
            f"Features: {len(FEATURES)}"
        )

    with info2:

        st.subheader(
            "Technology Stack"
        )

        st.write(
            "Python"
        )

        st.write(
            "Streamlit"
        )

        st.write(
            "Pandas"
        )

        st.write(
            "NumPy"
        )

        st.write(
            "Scikit-learn"
        )

        st.write(
            "Plotly"
        )

        st.write(
            "Random Forest"
        )

        st.write(
            "One-hot encoding"
        )

        st.write(
            "StandardScaler"
        )

        st.write(
            "Automated missing-value handling"
        )


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="custom-footer">

    <span class="footer-brand">
        CreditIQ
    </span>

    &nbsp;•&nbsp;

    Credit Approval Intelligence

    &nbsp;•&nbsp;

    Data Analytics & Machine Learning Project

</div>
""")