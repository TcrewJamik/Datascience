import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc
from mlxtend.plotting import plot_decision_regions
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.title("Исследование набора данных Anneal")

file_path = "anneal.data"

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, sep=",", header=None, na_values=["?"])
    data.columns = [
        "famiily",
        "product-type",
        "steel",
        "carbon",
        "hardness",
        "temper-rolling",
        "condition",
        "formability",
        "strength",
        "non-ageing",
        "surface-finish",
        "surface-quality",
        "enamelability",
        "bc",
        "bf",
        "bt",
        "bw/me",
        "bl",
        "m",
        "chrom",
        "phos",
        "cbond",
        "marvi",
        "exptl",
        "ferro",
        "corr",
        "blue/bright/varn/clean",
        "lustre",
        "jurofm",
        "s",
        "p",
        "shape",
        "thick",
        "width",
        "len",
        "oil",
        "bore",
        "packing",
        "class"
    ]
    return data

data = load_data(file_path)

st.header("Исходные данные")
st.dataframe(data)

st.header("Пропущенные значения (%) до обработки")
missing_percentage = data.isna().sum() / len(data) * 100
st.dataframe(pd.DataFrame({'Признак': missing_percentage.index, 'Процент пропусков': missing_percentage.values}))

data_cleaned = data.copy()
columns_to_drop = [
    "famiily", "temper-rolling", "non-ageing", "surface-finish", "enamelability",
    "bc", "bf", "bt", "bl", "m", "chrom", "phos", "cbond", "marvi", "exptl",
    "ferro", "corr", "blue/bright/varn/clean", "lustre", "jurofm", "s", "p",
    "oil", "packing", "bw/me"
]
data_cleaned = data_cleaned.drop(columns=columns_to_drop)

data_cleaned = data_cleaned.drop(columns=['carbon', 'hardness', 'strength', 'bore'])

data_cleaned = data_cleaned.dropna(subset=["class"])

class_counts = data_cleaned["class"].value_counts()

if len(data_cleaned["class"].unique()) > 2:
    total_samples = len(data_cleaned)
    dominant_class = class_counts.idxmax()
    dominant_count = class_counts.max()

    if dominant_count > total_samples / 2:
        data_cleaned["binary_class"] = data_cleaned["class"].apply(
            lambda x: 1 if x == dominant_class else 0
        )
    else:
        median_freq = class_counts.median()
        group1 = class_counts[class_counts >= median_freq].index.tolist()
        group2 = class_counts[class_counts < median_freq].index.tolist()

        data_cleaned["binary_class"] = data_cleaned["class"].apply(
            lambda x: 1 if x in group1 else 0
        )
else:
    data_cleaned["binary_class"] = data_cleaned["class"]

data_cleaned = data_cleaned.drop('class', axis = 1)
data_cleaned = data_cleaned.drop('product-type', axis = 1)

categorical_cols = data_cleaned.select_dtypes(include=['object']).columns

for col in categorical_cols:
    mode_value = data_cleaned[col].mode()[0]
    data_cleaned[col].fillna(mode_value, inplace=True)

median_formability = data_cleaned['formability'].median()
data_cleaned['formability'].fillna(median_formability, inplace=True)

label_encoder = LabelEncoder()
categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data_cleaned[col] = label_encoder.fit_transform(data_cleaned[col])


st.header("Очищенные данные")
st.dataframe(data_cleaned)

st.header("Распределение классов после обработки")
binary_class_counts = data_cleaned["binary_class"].value_counts()
st.bar_chart(binary_class_counts)

st.header("3D Scatter Plot (Топ 3 признака)")

feature_options = data_cleaned.columns.drop('binary_class').tolist()
feature_x = st.selectbox("Выберите признак для оси X", feature_options, index=feature_options.index('formability'))
feature_y = st.selectbox("Выберите признак для оси Y", feature_options, index=feature_options.index('condition'))
feature_z = st.selectbox("Выберите признак для оси Z", feature_options, index=feature_options.index('surface-quality'))

fig_scatter = plt.figure(figsize=(10, 8))
ax_scatter = fig_scatter.add_subplot(111, projection='3d')

colors = {0: 'red', 1: 'blue'}
markers = {0: 'o', 1: 's'}

for binary_class in data_cleaned['binary_class'].unique():
    subset = data_cleaned[data_cleaned['binary_class'] == binary_class]
    ax_scatter.scatter(subset[feature_x], subset[feature_y], subset[feature_z],
               c=[colors[binary_class]], marker=markers[binary_class], label=f'Class {binary_class}')

ax_scatter.set_xlabel(feature_x.capitalize())
ax_scatter.set_ylabel(feature_y.capitalize())
ax_scatter.set_zlabel(feature_z.capitalize())
ax_scatter.set_title('3D Scatter Plot of Anneal Dataset', fontsize=14)
ax_scatter.legend(title='Binary Class', fontsize=12, title_fontsize=12)

st.pyplot(fig_scatter)

st.header("Оценка моделей машинного обучения (Топ 2 признака)")

X = data_cleaned.drop('binary_class', axis=1)
y = data_cleaned['binary_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
numerical_cols = X_train.columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

top_2_features = ['formability', 'condition']
X_train_top2 = X_train[top_2_features]
X_test_top2 = X_test[top_2_features]

# KNN
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_top2, y_train)
y_pred_knn = knn_classifier.predict(X_test_top2)

# Logistic Regression
logistic_regression_classifier = LogisticRegression(max_iter=565, random_state=42, class_weight='balanced')
logistic_regression_classifier.fit(X_train_top2, y_train)
y_pred_logreg = logistic_regression_classifier.predict(X_test_top2)

# Decision Tree
decision_tree_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
decision_tree_classifier.fit(X_train_top2, y_train)
y_pred_dt = decision_tree_classifier.predict(X_test_top2)

st.subheader("Метрики производительности моделей")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("K-Nearest Neighbors")
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_knn):.3f}")
    st.metric("ROC AUC", f"{roc_auc_score(y_test, knn_classifier.predict_proba(X_test_top2)[:, 1]):.3f}")
    st.metric("F1", f"{f1_score(y_test, y_pred_knn):.3f}")
    st.metric("Precision", f"{precision_score(y_test, y_pred_knn):.3f}")
    st.metric("Recall", f"{recall_score(y_test, y_pred_knn):.3f}")

with col2:
    st.write("Logistic Regression")
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_logreg):.3f}")
    st.metric("ROC AUC", f"{roc_auc_score(y_test, logistic_regression_classifier.predict_proba(X_test_top2)[:, 1]):.3f}")
    st.metric("F1", f"{f1_score(y_test, y_pred_logreg):.3f}")
    st.metric("Precision", f"{precision_score(y_test, y_pred_logreg):.3f}")
    st.metric("Recall", f"{recall_score(y_test, y_pred_logreg):.3f}")

with col3:
    st.write("Decision Tree")
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_dt):.3f}")
    st.metric("ROC AUC", f"{roc_auc_score(y_test, decision_tree_classifier.predict_proba(X_test_top2)[:, 1]):.3f}")
    st.metric("F1", f"{f1_score(y_test, y_pred_dt):.3f}")
    st.metric("Precision", f"{precision_score(y_test, y_pred_dt):.3f}")
    st.metric("Recall", f"{recall_score(y_test, y_pred_dt):.3f}")

st.header("Границы решений")

X_train_top2_np = X_train_top2.values
y_train_np = y_train.values

# KNN Decision Boundary
knn_classifier_vis = KNeighborsClassifier(n_neighbors=3)
knn_classifier_vis.fit(X_train_top2_np, y_train_np)
fig_knn_db = plt.figure(figsize=(10, 8))
plot_decision_regions(X_train_top2_np, y_train_np, clf=knn_classifier_vis, legend=2)
plt.xlabel(top_2_features[0].capitalize())
plt.ylabel(top_2_features[1].capitalize())
plt.title('KNN Decision Boundary (k=3)')
plt.legend(title='Binary Class')
plt.grid(True)
st.pyplot(fig_knn_db)

# Logistic Regression Decision Boundary
logistic_regression_classifier_vis = LogisticRegression(max_iter=565, random_state=42)
logistic_regression_classifier_vis.fit(X_train_top2_np, y_train_np)
fig_logreg_db = plt.figure(figsize=(10, 8))
plot_decision_regions(X_train_top2_np, y_train_np, clf=logistic_regression_classifier_vis, legend=2)
plt.xlabel(top_2_features[0].capitalize())
plt.ylabel(top_2_features[1].capitalize())
plt.title('Logistic Regression Decision Boundary')
plt.legend(title='Binary Class')
plt.grid(True)
st.pyplot(fig_logreg_db)

# Decision Tree Decision Boundary
decision_tree_classifier_vis = DecisionTreeClassifier(max_depth=5, random_state=42)
decision_tree_classifier_vis.fit(X_train_top2_np, y_train_np)
fig_dt_db = plt.figure(figsize=(10, 8))
plot_decision_regions(X_train_top2_np, y_train_np, clf=decision_tree_classifier_vis, legend=2)
plt.xlabel(top_2_features[0].capitalize())
plt.ylabel(top_2_features[1].capitalize())
plt.title('Decision Tree Decision Boundary (Max Depth=5)')
plt.legend(title='Binary Class')
plt.grid(True)
st.pyplot(fig_dt_db)

st.header("ROC-кривые")

y_prob_knn = knn_classifier.predict_proba(X_test_top2)[:, 1]
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_prob_knn)
auc_knn = auc(fpr_knn, tpr_knn)

y_prob_logreg = logistic_regression_classifier.predict_proba(X_test_top2)[:, 1]
fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_test, y_prob_logreg)
auc_logreg = auc(fpr_logreg, tpr_logreg)

y_prob_dt = decision_tree_classifier_vis.predict_proba(X_test_top2)[:, 1]
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_prob_dt)
auc_dt = auc(fpr_dt, tpr_dt)

fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
ax_roc.plot(fpr_knn, tpr_knn, label=f'K-Nearest Neighbors (AUC = {auc_knn:.2f})')
ax_roc.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {auc_logreg:.2f})')
ax_roc.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})')
ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Guess')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC-кривые для разных классификаторов')
ax_roc.legend(title='Классификатор')
ax_roc.set_xlim([0, 1])
ax_roc.set_ylim([0, 1])
ax_roc.grid(True)
st.pyplot(fig_roc)

st.header("AUC на трейне и тесте")

col4, col5, col6 = st.columns(3)

with col4:
    st.write("K-Nearest Neighbors")
    y_prob_train_knn = knn_classifier_vis.predict_proba(X_train_top2)[:, 1]
    auc_train_knn = auc(roc_curve(y_train, y_prob_train_knn)[0], roc_curve(y_train, y_prob_train_knn)[1])
    y_prob_test_knn = knn_classifier_vis.predict_proba(X_test_top2)[:, 1]
    auc_test_knn = auc(roc_curve(y_test, y_prob_test_knn)[0], roc_curve(y_test, y_prob_test_knn)[1])
    st.metric("AUC на трейне", f"{auc_train_knn:.2f}")
    st.metric("AUC на тесте", f"{auc_test_knn:.2f}")

with col5:
    st.write("Logistic Regression")
    y_prob_train_logreg = logistic_regression_classifier_vis.predict_proba(X_train_top2)[:, 1]
    auc_train_logreg = auc(roc_curve(y_train, y_prob_train_logreg)[0], roc_curve(y_train, y_prob_train_logreg)[1])
    y_prob_test_logreg = logistic_regression_classifier_vis.predict_proba(X_test_top2)[:, 1]
    auc_test_logreg = auc(roc_curve(y_test, y_prob_test_logreg)[0], roc_curve(y_test, y_prob_test_logreg)[1])
    st.metric("AUC на трейне", f"{auc_train_logreg:.2f}")
    st.metric("AUC на тесте", f"{auc_test_logreg:.2f}")

with col6:
    st.write("Decision Tree")
    y_prob_train_dt = decision_tree_classifier_vis.predict_proba(X_train_top2)[:, 1]
    auc_train_dt = auc(roc_curve(y_train, y_prob_train_dt)[0], roc_curve(y_train, y_prob_train_dt)[1])
    y_prob_test_dt = decision_tree_classifier_vis.predict_proba(X_test_top2)[:, 1]
    auc_test_dt = auc(roc_curve(y_test, y_prob_test_dt)[0], roc_curve(y_test, y_prob_test_dt)[1])
    st.metric("AUC на трейне", f"{auc_train_dt:.2f}")
    st.metric("AUC на тесте", f"{auc_test_dt:.2f}")
