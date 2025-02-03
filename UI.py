import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_sample_weight
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import json
from datetime import datetime

# Initialize session state for benchmarks if not exists
if 'benchmarks' not in st.session_state:
    st.session_state.benchmarks = []

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

def save_benchmark(model_name, params, metrics):
    """Save current model configuration and results to benchmarks"""
    benchmark = {
        'model_name': model_name,
        'parameters': params,
        'metrics': metrics
    }
    st.session_state.benchmarks.append(benchmark)

def display_benchmarks():
    """Display all saved benchmarks in a comparative view"""
    if not st.session_state.benchmarks:
        st.info("No benchmarks saved yet. Run some models to add to benchmarks!")
        return

    st.subheader("Benchmarks Comparison")
    
    # Convert benchmarks to DataFrame for easier display
    benchmark_records = []
    for b in st.session_state.benchmarks:
        # Create parameter string
        param_str = ", ".join([f"{k}={v}" for k, v in b['parameters'].items()])
        
        record = {
            'Parameters': param_str,
            'Model': b['model_name'],
            'Accuracy': b['metrics']['Accuracy'],
            'Precision': b['metrics']['Precision'],
            'Recall': b['metrics']['Recall'],
            'F1 Score': b['metrics']['F1 Score']
        }
        benchmark_records.append(record)
    
    df_benchmarks = pd.DataFrame(benchmark_records)
    
    # Display benchmarks table
    st.dataframe(df_benchmarks)
    
    # Plot comparative metrics
    st.subheader("Comparative Performance")
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']  # Green, Blue, Red, Yellow
    
    x = np.arange(len(df_benchmarks))
    width = 0.2
    multiplier = 0
    
    for metric, color in zip(metrics_to_plot, colors):
        offset = width * multiplier
        ax.bar(x + offset, df_benchmarks[metric], width, label=metric, color=color)
        multiplier += 1
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{row['Model']}\n({row['Parameters']})" for _, row in df_benchmarks.iterrows()], rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    st.pyplot(fig)

    # Add option to clear benchmarks
    if st.button("Clear All Benchmarks"):
        st.session_state.benchmarks = []
        st.rerun()

# Streamlit app configuration
st.set_page_config(page_title="Model Performans Analizi", layout="wide")
st.title("Model Performansı Görselleştirme")

# Veri yükleme seçeneği
st.sidebar.header("Veri Yükleme")
data_source = st.sidebar.radio(
    "Veri seti kaynağını seçin:",
    ("Local", "Yükle")
)

# Veri yükleme
if data_source == "Local":
    data_path = st.sidebar.text_input("Local veri seti yolunu girin:", value="./dataset/synthetic_covid_impact_on_work.csv")
    if not os.path.exists(data_path):
        st.sidebar.error("Belirtilen dosya yolu bulunamadı. Lütfen doğru bir yol girin.")
        st.stop()
    raw_data = pd.read_csv(data_path)
elif data_source == "Yükle":
    uploaded_file = st.sidebar.file_uploader("Veri setinizi yükleyin (CSV formatında):", type="csv")
    if uploaded_file is not None:
        raw_data = pd.read_csv(uploaded_file)
    else:
        st.sidebar.warning("Lütfen bir dosya yükleyin.")
        st.stop()

# Process data
data = raw_data.copy()

# Check if target variable exists
if 'Productivity_Change' not in data.columns:
    st.error(f"Hedef değişken (Productivity_Change) veride bulunamadı!")
    st.stop()
    
# Get column types
cat_cols, num_cols, cat_but_car = grab_col_names(data)

# Separate target variable
target = data['Productivity_Change'].copy()
data = data.drop(columns=['Productivity_Change'])

# Handle categorical variables
for col in cat_cols:
    if col != 'Productivity_Change':
        if data[col].nunique() <= 10:
            data = pd.get_dummies(data, columns=[col], prefix=col)
        else:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

# Handle numerical variables
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    # Handle outliers
    low_limit, up_limit = outlier_thresholds(data, col, q1=0.01, q3=0.99)
    data[col] = data[col].clip(lower=low_limit, upper=up_limit)

# Scale features
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data[numeric_cols])
poly_feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
data = pd.concat([
    data,
    pd.DataFrame(poly_features[:, len(numeric_cols):], 
                columns=poly_feature_names[len(numeric_cols):])
], axis=1)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=20)
selected_features = selector.fit_transform(data, target)
selected_cols = data.columns[selector.get_support()].tolist()
data = pd.DataFrame(selected_features, columns=selected_cols)

# Add back target
le = LabelEncoder()
data['Productivity_Change'] = le.fit_transform(target)

# Model selection
st.sidebar.header("Model ve Hiperparametre Ayarları")
algorithm = st.sidebar.selectbox("Bir algoritma seçin:", ["Logistic Regression", "Random Forest", "XGBoost"])

if algorithm == "Logistic Regression":
    C = st.sidebar.number_input("C (Regularization Strength):", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    solver = st.sidebar.selectbox("Solver:", ["lbfgs", "liblinear", "saga"])
    max_iter = st.sidebar.number_input("Max Iterations:", min_value=50, max_value=500, value=100, step=10)
    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
elif algorithm == "Random Forest":
    n_estimators = st.sidebar.number_input("Number of Trees (n_estimators):", min_value=10, max_value=1000, value=200, step=10)
    max_depth = st.sidebar.number_input("Max Depth:", min_value=1, max_value=50, value=15, step=1)
    min_samples_split = st.sidebar.number_input("Min Samples Split:", min_value=2, max_value=20, value=5, step=1)
    min_samples_leaf = st.sidebar.number_input("Min Samples Leaf:", min_value=1, max_value=10, value=2, step=1)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
elif algorithm == "XGBoost":
    n_estimators = st.sidebar.number_input("Number of Trees (n_estimators):", min_value=10, max_value=500, value=100, step=10)
    learning_rate = st.sidebar.number_input("Learning Rate:", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    max_depth = st.sidebar.number_input("Max Depth:", min_value=1, max_value=50, value=6, step=1)
    model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, use_label_encoder=False)

# Before model training, prepare X and y
X = data.drop(columns=['Productivity_Change'])
y = data['Productivity_Change']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model and get predictions
if algorithm == "XGBoost":
    model.fit(X_train, y_train)
else:
    # For other models, use balanced sample weights
    sample_weights = compute_sample_weight('balanced', y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

# Calculate metrics
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average='weighted'),
    "Recall": recall_score(y_test, y_pred, average='weighted'),
    "F1 Score": f1_score(y_test, y_pred, average='weighted')
}

if y_proba is not None and len(np.unique(y_test)) == 2:
    metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba[:, 1])

# Display metrics
st.subheader("Model Metrikleri")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
col2.metric("Precision", f"{metrics['Precision']:.3f}")
col3.metric("Recall", f"{metrics['Recall']:.3f}")
col4.metric("F1 Score", f"{metrics['F1 Score']:.3f}")

# Plot confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
st.pyplot(plt)

# Display feature importance if available
if hasattr(model, "feature_importances_"):
    st.subheader("Feature Importance")
    # Get feature names from X (not data) to match feature_importances_ length
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances.head(10))
    plt.title("Top 10 Most Important Features")
    st.pyplot(plt)

# Benchmark controls and saving
st.sidebar.header("Benchmark Controls")
save_to_benchmark = st.sidebar.checkbox("Save Results to Benchmark")
show_benchmarks = st.sidebar.checkbox("Show Benchmarks")

if save_to_benchmark and 'metrics' in locals():
    # Get current model parameters
    if algorithm == "Logistic Regression":
        params = {'C': C, 'solver': solver, 'max_iter': max_iter}
    elif algorithm == "Random Forest":
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }
    elif algorithm == "XGBoost":
        params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth}
    
    # Save benchmark
    save_benchmark(
        model_name=algorithm,
        params=params,
        metrics=metrics
    )
    st.success("Results saved to benchmarks!")

if show_benchmarks:
    display_benchmarks()