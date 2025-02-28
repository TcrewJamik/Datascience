import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, roc_curve, auc)
from mlxtend.plotting import plot_decision_regions
import warnings
warnings.filterwarnings('ignore')

st.title("Анализ данных Anneal с использованием Streamlit")

# Загрузка и первичная обработка данных
@st.cache(allow_output_mutation=True)
def load_data():
    file_path = "anneal.data"  # убедитесь, что файл находится в рабочей директории
    data = pd.read_csv(file_path, sep=",", header=None, na_values=["?"])
    data.columns = [
        "famiily", "product-type", "steel", "carbon", "hardness", "temper-rolling", "condition",
        "formability", "strength", "non-ageing", "surface-finish", "surface-quality", "enamelability",
        "bc", "bf", "bt", "bw/me", "bl", "m", "chrom", "phos", "cbond", "marvi", "exptl",
        "ferro", "corr", "blue/bright/varn/clean", "lustre", "jurofm", "s", "p", "shape", "thick",
        "width", "len", "oil", "bore", "packing", "class"
    ]
    return data

data = load_data()
st.subheader("Исходные данные")
st.dataframe(data.head())

# Вычисление процента пропусков по столбцам
missing_pct = data.isna().sum() / len(data) * 100
st.write("Процент пропущенных значений по столбцам:")
st.write(missing_pct)

# Удаление ненужных столбцов
cols_to_drop = [
    "famiily", "temper-rolling", "non-ageing", "surface-finish", "enamelability",
    "bc", "bf", "bt", "bl", "m", "chrom", "phos", "cbond", "marvi", "exptl",
    "ferro", "corr", "blue/bright/varn/clean", "lustre", "jurofm", "s", "p",
    "oil", "packing", "bw/me"
]
data = data.drop(columns=cols_to_drop)
data = data.drop(columns=['carbon', 'hardness', 'strength', 'bore'])

st.subheader("Данные после удаления столбцов")
st.dataframe(data.head())

# Обработка целевого столбца: создание бинарного класса
data = data.dropna(subset=["class"])
class_counts = data["class"].value_counts()
st.write("Распределение классов до объединения:")
st.write(class_counts)

unique_classes = data["class"].unique()
if len(unique_classes) > 2:
    total_samples = len(data)
    dominant_class = class_counts.idxmax()
    dominant_count = class_counts.max()
    if dominant_count > total_samples / 2:
        data["binary_class"] = data["class"].apply(lambda x: dominant_class if x == dominant_class else "others")
    else:
        median_freq = class_counts.median()
        group1 = class_counts[class_counts >= median_freq].index.tolist()
        group2 = class_counts[class_counts < median_freq].index.tolist()
        data["binary_class"] = data["class"].apply(lambda x: "group1" if x in group1 else "group2")
else:
    data["binary_class"] = data["class"]

st.write("Распределение бинарного класса:")
st.write(data["binary_class"].value_counts())

# Удаление исходного столбца 'class'
data = data.drop('class', axis=1)

st.write("Количество уникальных значений по столбцам:")
st.write(data.nunique())
st.write("Категориальные признаки:")
st.write(data.select_dtypes(include=['object']).columns)

# Заполнение пропусков в категориальных столбцах наиболее частыми значениями (mode)
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_value = data[col].mode()[0]
    data[col].fillna(mode_value, inplace=True)

st.subheader("Данные после заполнения категориальных пропусков")
st.dataframe(data.head())
st.write("Количество уникальных значений по столбцам:")
st.write(data.nunique())

# Удаление столбца 'product-type'
data = data.drop('product-type', axis=1)

# Преобразование категориальных признаков в числовые (Label Encoding)
categorical_cols = data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

st.subheader("Данные после кодирования категориальных признаков")
st.dataframe(data.head())

# Заполнение пропусков для 'formability' медианным значением
median_formability = data['formability'].median()
data['formability'].fillna(median_formability, inplace=True)

st.write("Пропуски после обработки:")
st.write(data.isna().sum())

# Корреляционный анализ с бинарным классом
correlations = data.corr()['binary_class'].drop('binary_class')
top_3_features = correlations.abs().nlargest(3)
st.write("Топ 3 признака с наилучшей корреляцией с таргетом:")
for feature_name, correlation_value in top_3_features.items():
    st.write(f"- Признак: '{feature_name}', Корреляция: {correlation_value:.3f}")

# 3D-график (до масштабирования) для трёх признаков: formability, condition, surface-quality
st.subheader("3D Scatter Plot (До масштабирования)")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Подбираем цвета и маркеры (если классов больше 2, можно добавить новые)
colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
markers = {0: 'o', 1: 's', 2: '^', 3: 'D'}

for cls in np.unique(data['binary_class']):
    subset = data[data['binary_class'] == cls]
    ax.scatter(subset['formability'], subset['condition'], subset['surface-quality'],
               c=colors.get(cls, 'black'), marker=markers.get(cls, 'o'), label=f'Class {cls}')
ax.set_xlabel('Formability')
ax.set_ylabel('Condition')
ax.set_zlabel('Surface Quality')
ax.set_title('3D Scatter Plot до масштабирования')
ax.legend(title='Binary Class')
st.pyplot(fig)

# Разбиение на обучающую и тестовую выборки
X = data.drop('binary_class', axis=1)
y = data['binary_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
numerical_cols = X_train.columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Используем только два признака для визуализации и классификации
top_2_features = ['formability', 'condition']
X_train_top2 = X_train[top_2_features]
X_test_top2 = X_test[top_2_features]

st.subheader("Обучение моделей (на топ-2 признаках)")

# K-Nearest Neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_top2, y_train)
y_pred_knn = knn_classifier.predict(X_test_top2)
st.write("### K-Nearest Neighbors (K=3)")
st.write("Accuracy:", accuracy_score(y_test, y_pred_knn))
st.write("ROC AUC:", roc_auc_score(y_test, knn_classifier.predict_proba(X_test_top2)[:, 1]))
st.write("F1 Score:", f1_score(y_test, y_pred_knn))
st.write("Precision:", precision_score(y_test, y_pred_knn))
st.write("Recall:", recall_score(y_test, y_pred_knn))

# Logistic Regression
logistic_regression_classifier = LogisticRegression(max_iter=565, random_state=42, class_weight='balanced')
logistic_regression_classifier.fit(X_train_top2, y_train)
y_pred_logreg = logistic_regression_classifier.predict(X_test_top2)
st.write("### Logistic Regression")
st.write("Accuracy:", accuracy_score(y_test, y_pred_logreg))
st.write("ROC AUC:", roc_auc_score(y_test, logistic_regression_classifier.predict_proba(X_test_top2)[:, 1]))
st.write("F1 Score:", f1_score(y_test, y_pred_logreg))
st.write("Precision:", precision_score(y_test, y_pred_logreg))
st.write("Recall:", recall_score(y_test, y_pred_logreg))

# Decision Tree
decision_tree_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
decision_tree_classifier.fit(X_train_top2, y_train)
y_pred_dt = decision_tree_classifier.predict(X_test_top2)
st.write("### Decision Tree (Max Depth=5)")
st.write("Accuracy:", accuracy_score(y_test, y_pred_dt))
st.write("ROC AUC:", roc_auc_score(y_test, decision_tree_classifier.predict_proba(X_test_top2)[:, 1]))
st.write("F1 Score:", f1_score(y_test, y_pred_dt))
st.write("Precision:", precision_score(y_test, y_pred_dt))
st.write("Recall:", recall_score(y_test, y_pred_dt))

# Визуализация границ решений для каждой модели
st.subheader("Визуализация границ решений")

# Функция для отображения границ решений с помощью mlxtend
def plot_model_decision_boundary(clf, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_decision_regions(X_train_top2.values, y_train.values, clf=clf)
    plt.xlabel(top_2_features[0].capitalize())
    plt.ylabel(top_2_features[1].capitalize())
    plt.title(title)
    plt.grid(True)
    st.pyplot(fig)

plot_model_decision_boundary(KNeighborsClassifier(n_neighbors=3), 'KNN Decision Boundary (k=3)')
plot_model_decision_boundary(LogisticRegression(max_iter=565, random_state=42), 'Logistic Regression Decision Boundary')
plot_model_decision_boundary(DecisionTreeClassifier(max_depth=5, random_state=42), 'Decision Tree Decision Boundary (Max Depth=5)')

# Построение ROC-кривых для всех моделей
st.subheader("ROC-кривые моделей")
y_prob_knn = knn_classifier.predict_proba(X_test_top2)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
auc_knn = auc(fpr_knn, tpr_knn)

y_prob_logreg = logistic_regression_classifier.predict_proba(X_test_top2)[:, 1]
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)
auc_logreg = auc(fpr_logreg, tpr_logreg)

y_prob_dt = decision_tree_classifier.predict_proba(X_test_top2)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
auc_dt = auc(fpr_dt, tpr_dt)

fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {auc_knn:.2f})')
plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {auc_logreg:.2f})')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые для разных классификаторов')
plt.legend(title='Классификатор')
plt.grid(True)
st.pyplot(fig_roc)

# Вычисление AUC для трейна и теста по каждой модели
st.subheader("AUC на трейне и тесте")

# KNN
y_prob_train_knn = knn_classifier.predict_proba(X_train_top2)[:, 1]
fpr_train_knn, tpr_train_knn, _ = roc_curve(y_train, y_prob_train_knn)
auc_train_knn = auc(fpr_train_knn, tpr_train_knn)
st.write("K-Nearest Neighbors (K=3):")
st.write(f"AUC на трейне: {auc_train_knn:.2f}")
st.write(f"AUC на тесте: {auc_knn:.2f}")

# Logistic Regression
y_prob_train_logreg = logistic_regression_classifier.predict_proba(X_train_top2)[:, 1]
fpr_train_logreg, tpr_train_logreg, _ = roc_curve(y_train, y_prob_train_logreg)
auc_train_logreg = auc(fpr_train_logreg, tpr_train_logreg)
st.write("Logistic Regression:")
st.write(f"AUC на трейне: {auc_train_logreg:.2f}")
st.write(f"AUC на тесте: {auc_logreg:.2f}")

# Decision Tree
y_prob_train_dt = decision_tree_classifier.predict_proba(X_train_top2)[:, 1]
fpr_train_dt, tpr_train_dt, _ = roc_curve(y_train, y_prob_train_dt)
auc_train_dt = auc(fpr_train_dt, tpr_train_dt)
st.write("Decision Tree (Max Depth=5):")
st.write(f"AUC на трейне: {auc_train_dt:.2f}")
st.write(f"AUC на тесте: {auc_dt:.2f}")
