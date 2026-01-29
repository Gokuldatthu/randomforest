import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Random Forest Explorer", layout="wide")
st.title("Random Forest Explorer")

st.markdown("Upload a CSV or use the sample `RF datasets.csv` in the folder.")

# Data load
uploaded = st.file_uploader("Upload CSV", type=["csv"] )
use_sample = st.checkbox("Use sample file (RF datasets.csv)", value=True)

@st.cache_data
def load_csv_from_path(path):
    return pd.read_csv(path)

@st.cache_data
def load_uploaded(f):
    return pd.read_csv(f)

if uploaded is not None:
    df = load_uploaded(uploaded)
else:
    if use_sample:
        try:
            df = load_csv_from_path("RF datasets.csv")
        except Exception:
            try:
                df = load_csv_from_path("./RF datasets.csv")
            except Exception:
                df = None
    else:
        df = None

if df is None:
    st.warning("No dataset loaded yet. Upload a CSV or enable the sample file.")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head())

# Select target and features
cols = df.columns.tolist()
target = st.selectbox("Select target column", options=cols)
features = st.multiselect("Select feature columns (leave to auto-select all except target)", options=[c for c in cols if c!=target], default=[c for c in cols if c!=target])

if not features:
    st.error("Pick at least one feature column.")
    st.stop()

# Quick EDA
with st.expander("Dataset info and EDA"):
    st.write(df.describe(include='all'))
    st.write("Missing values per column:")
    st.write(df.isnull().sum())
    if len(features) <= 10:
        st.subheader("Pairplot (may be slow)")
        try:
            fig = sns.pairplot(df[features + [target]].dropna())
            st.pyplot(fig)
        except Exception:
            st.write("Pairplot failed or too big for display.")

# Preprocessing: simple dropna
df2 = df[features + [target]].dropna()

# Determine classification vs regression
is_classification = pd.api.types.is_numeric_dtype(df2[target]) and (df2[target].nunique() <= 20) == False
# Better heuristic: if dtype is object or few uniques -> classification
if df2[target].dtype == object or df2[target].nunique() < 20:
    task = 'classification'
else:
    task = 'regression'

st.info(f"Detected task type: {task}")

# Train-test split UI
test_size = st.slider("Test set fraction", 0.05, 0.5, 0.2)
random_state = st.number_input("Random seed", value=42, step=1)

X = df2[features]
y = df2[target]

# Convert categorical features to dummies (simple)
X_processed = pd.get_dummies(X, drop_first=True)

if st.button("Train Random Forest"):
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, random_state=int(random_state))

    if task == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=int(random_state))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.metric("Accuracy", f"{acc:.4f}")
        st.subheader("Classification report")
        st.text(classification_report(y_test, preds))
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        # Probabilities
        if hasattr(model, "predict_proba"):
            st.subheader("Prediction probabilities sample")
            st.write(pd.DataFrame(model.predict_proba(X_test), columns=model.classes_).head())

    else:
        model = RandomForestRegressor(n_estimators=100, random_state=int(random_state))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("R2", f"{r2:.4f}")

    # Feature importances
    st.subheader("Feature importances")
    importances = pd.Series(model.feature_importances_, index=X_processed.columns).sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(6, max(4, 0.2*len(importances))))
    sns.barplot(x=importances.values[:30], y=importances.index[:30], ax=ax2)
    st.pyplot(fig2)

    # Save model into session state for prediction
    st.session_state['rf_model'] = model
    st.session_state['X_columns'] = X_processed.columns.tolist()
    st.success("Model trained and stored in session. Use Prediction panel below.")

# Prediction UI
st.subheader("Make a prediction")
if 'rf_model' in st.session_state:
    model = st.session_state['rf_model']
    cols_used = st.session_state['X_columns']

    # Build input form dynamically for numeric features only (simple)
    st.write("Provide values for features (missing dummy handling is automatic):")
    input_vals = {}
    for f in features:
        # attempt to infer numeric or categorical
        if pd.api.types.is_numeric_dtype(df[f]):
            val = st.number_input(f, value=float(df[f].median()))
            input_vals[f] = val
        else:
            opts = df[f].dropna().unique().tolist()
            if not opts:
                opts = ['']
            val = st.selectbox(f, options=opts)
            input_vals[f] = val

    if st.button("Predict"):
        single = pd.DataFrame([input_vals])
        single_proc = pd.get_dummies(single, drop_first=True)
        # align to training columns
        for c in cols_used:
            if c not in single_proc.columns:
                single_proc[c] = 0
        single_proc = single_proc[cols_used]
        pred = model.predict(single_proc)[0]
        st.write("Prediction:", pred)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(single_proc)[0]
            proba_df = pd.DataFrame([proba], columns=model.classes_)
            st.write("Probabilities:")
            st.write(proba_df.T.sort_values(0, ascending=False).head())
else:
    st.info("Train a model first to enable predictions.")

st.sidebar.caption("Streamlit Random Forest Explorer — minimal demo\nDependencies: pandas, scikit-learn, streamlit, seaborn, matplotlib")
