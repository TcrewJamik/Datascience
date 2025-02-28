import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Исследование Anneal Dataset", page_icon="📈", layout="wide")

# ---- Загрузка данных ----
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

data_original = load_data(file_path) # Keep original data for display

data = data_original.copy() # Work with a copy for cleaning

# ---- Предобработка данных ----
columns_to_drop = [
    "famiily", "temper-rolling", "non-ageing", "surface-finish", "enamelability",
    "bc", "bf", "bt", "bl", "m", "chrom", "phos", "cbond", "marvi", "exptl",
    "ferro", "corr", "blue/bright/varn/clean", "lustre", "jurofm", "s", "p",
    "oil", "packing", "bw/me"
]
data.drop(columns=columns_to_drop, inplace=True)
data.drop(columns=['carbon', 'hardness', 'strength', 'bore', 'product-type'], inplace=True)
data.dropna(subset=["class"], inplace=True)

class_counts = data["class"].value_counts()
if len(data["class"].unique()) > 2:
    median_freq = class_counts.median()
    group1 = class_counts[class_counts >= median_freq].index.tolist()
    data["binary_class"] = data["class"].apply(lambda x: 1 if x in group1 else 0)
else:
    data["binary_class"] = data["class"]
data.drop('class', axis=1, inplace=True)

categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_value = data[col].mode()[0]
    data[col].fillna(mode_value, inplace=True)
median_formability = data['formability'].median()
data['formability'].fillna(median_formability, inplace=True)

label_encoder = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# ---- Разделение данных ----
X = data.drop('binary_class', axis=1)
y = data['binary_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
numerical_cols = X_train.columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# ---- Streamlit App Layout ----
st.title("Исследование набора данных Anneal")
st.markdown("Добро пожаловать в интерактивное исследование набора данных Anneal! "
            "Изучите данные, визуализируйте их и оцените работу различных моделей машинного обучения.")

# ---- Sidebar for Controls ----
with st.sidebar:
    st.header("Настройки моделей")
    n_neighbors_knn = st.slider("Количество соседей для KNN", min_value=1, max_value=20, value=3, step=1)
    max_depth_dt = st.slider("Максимальная глубина дерева решений", min_value=1, max_value=10, value=5, step=1)
    available_features = X_train.columns.tolist()
    default_features = ['formability', 'condition'] if all(f in available_features for f in ['formability', 'condition']) else available_features[:min(2, len(available_features))] # Default to first 2 if defaults not available
    selected_features = st.multiselect("Выберите признаки для обучения моделей", available_features, default=default_features)
    show_decision_boundaries = st.checkbox("Показать границы решений", value=True)
    retrain_button = st.button("Переобучить модели")

# ---- Data Exploration Section ----
st.header("Обзор данных")
st.subheader("Исходные данные")
st.dataframe(data_original.head()) # Display first few rows of original data for context

st.subheader("Описательная статистика")
st.dataframe(data.describe())

st.subheader("Распределение классов (бинарная целевая переменная)")
binary_class_counts = data["binary_class"].value_counts()
st.bar_chart(binary_class_counts)

# Feature Histograms
st.subheader("Гистограммы признаков")
feature_hist_cols = st.columns(3) # Display histograms in columns for better layout
for i, col in enumerate(X_train.columns):
    with feature_hist_cols[i % 3]:
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(data=X_train, x=col, kde=True, ax=ax_hist)
        st.pyplot(fig_hist)

# ---- 3D Scatter Plot ----
if len(selected_features) >= 3:
    st.header("3D Scatter Plot (Интерактивный)")
    feature_x_3d = st.selectbox("Выберите признак для оси X (3D Scatter)", selected_features, index=selected_features.index(selected_features[0]) if selected_features[0] in selected_features else 0)
    feature_y_3d = st.selectbox("Выберите признак для оси Y (3D Scatter)", selected_features, index=selected_features.index(selected_features[1]) if selected_features[1] in selected_features else 1)
    feature_z_3d = st.selectbox("Выберите признак для оси Z (3D Scatter)", selected_features, index=selected_features.index(selected_features[2]) if selected_features[2] in selected_features else 2)

    fig_scatter_plotly = px.scatter_3d(data, x=feature_x_3d, y=feature_y_3d, z=feature_z_3d, color='binary_class',
                                     symbol='binary_class', labels={feature_x_3d: feature_x_3d.capitalize(),
                                                                   feature_y_3d: feature_y_3d.capitalize(),
                                                                   feature_z_3d: feature_z_3d.capitalize(),
                                                                   'binary_class': 'Binary Class'})
    fig_scatter_plotly.update_layout(title_text='3D Scatter Plot (Plotly)')
    st.plotly_chart(fig_scatter_plotly)
else:
    st.info("Выберите как минимум три признака в боковой панели для отображения 3D Scatter Plot.")

# ---- Model Training and Evaluation ----
if retrain_button or not st.session_state.get('models_trained', False): # Retrain if button pressed or models not trained yet
    st.session_state['models_trained'] = True # Mark models as trained

    if len(selected_features) < 2:
        st.warning("Для обучения моделей и визуализации границ решений выберите как минимум два признака в боковой панели. Модели обучены на дефолтных признаках.")
        X_train_selected = X_train[default_features[:min(2, len(default_features))]] # Use defaults if less than 2 selected
        X_test_selected = X_test[default_features[:min(2, len(default_features))]]
    else:
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]


    # KNN Model
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors_knn)
    knn_classifier.fit(X_train_selected, y_train)
    y_pred_knn = knn_classifier.predict(X_test_selected)
    y_prob_knn = knn_classifier.predict_proba(X_test_selected)[:, 1]

    # Logistic Regression Model
    logistic_regression_classifier = LogisticRegression(max_iter=565, random_state=42, class_weight='balanced')
    logistic_regression_classifier.fit(X_train_selected, y_train)
    y_pred_logreg = logistic_regression_classifier.predict(X_test_selected)
    y_prob_logreg = logistic_regression_classifier.predict_proba(X_test_selected)[:, 1]

    # Decision Tree Model
    decision_tree_classifier = DecisionTreeClassifier(max_depth=max_depth_dt, random_state=42)
    decision_tree_classifier.fit(X_train_selected, y_train)
    y_pred_dt = decision_tree_classifier.predict(X_test_selected)
    y_prob_dt = decision_tree_classifier.predict_proba(X_test_selected)[:, 1]

    st.session_state['knn_classifier'] = knn_classifier # Store models for later use if needed (e.g., feature importance)
    st.session_state['logreg_classifier'] = logistic_regression_classifier
    st.session_state['dt_classifier'] = decision_tree_classifier
    st.session_state['X_train_selected'] = X_train_selected
    st.session_state['y_train'] = y_train
    st.session_state['X_test_selected'] = X_test_selected
    st.session_state['y_test'] = y_test
    st.session_state['y_pred_knn'] = y_pred_knn
    st.session_state['y_prob_knn'] = y_prob_knn
    st.session_state['y_pred_logreg'] = y_pred_logreg
    st.session_state['y_prob_logreg'] = y_prob_logreg
    st.session_state['y_pred_dt'] = y_pred_dt
    st.session_state['y_prob_dt'] = y_prob_dt


# ---- Model Evaluation Display ----
st.header("Оценка моделей машинного обучения")

if st.session_state.get('models_trained', False): # Only display evaluation if models are trained
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("K-Nearest Neighbors")
        st.metric("Accuracy", f"{accuracy_score(st.session_state['y_test'], st.session_state['y_pred_knn']):.3f}")
        st.metric("ROC AUC", f"{roc_auc_score(st.session_state['y_test'], st.session_state['y_prob_knn']):.3f}")
        st.metric("F1", f"{f1_score(st.session_state['y_test'], st.session_state['y_pred_knn']):.3f}")
        cm_knn = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred_knn'])
        fig_cm_knn, ax_cm_knn = plt.subplots()
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax_cm_knn)
        ax_cm_knn.set_xlabel('Предсказанные классы')
        ax_cm_knn.set_ylabel('Истинные классы')
        ax_cm_knn.set_title('Confusion Matrix')
        st.pyplot(fig_cm_knn)
        st.text("Classification Report:\n" + classification_report(st.session_state['y_test'], st.session_state['y_pred_knn']))

    with col2:
        st.subheader("Logistic Regression")
        st.metric("Accuracy", f"{accuracy_score(st.session_state['y_test'], st.session_state['y_pred_logreg']):.3f}")
        st.metric("ROC AUC", f"{roc_auc_score(st.session_state['y_test'], st.session_state['y_prob_logreg']):.3f}")
        st.metric("F1", f"{f1_score(st.session_state['y_test'], st.session_state['y_pred_logreg']):.3f}")
        cm_logreg = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred_logreg'])
        fig_cm_logreg, ax_cm_logreg = plt.subplots()
        sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Greens', ax=ax_cm_logreg)
        ax_cm_logreg.set_xlabel('Предсказанные классы')
        ax_cm_logreg.set_ylabel('Истинные классы')
        ax_cm_logreg.set_title('Confusion Matrix')
        st.pyplot(fig_cm_logreg)
        st.text("Classification Report:\n" + classification_report(st.session_state['y_test'], st.session_state['y_pred_logreg']))

    with col3:
        st.subheader("Decision Tree")
        st.metric("Accuracy", f"{accuracy_score(st.session_state['y_test'], st.session_state['y_pred_dt']):.3f}")
        st.metric("ROC AUC", f"{roc_auc_score(st.session_state['y_test'], st.session_state['y_prob_dt']):.3f}")
        st.metric("F1", f"{f1_score(st.session_state['y_test'], st.session_state['y_pred_dt']):.3f}")
        cm_dt = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred_dt'])
        fig_cm_dt, ax_cm_dt = plt.subplots()
        sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Reds', ax=ax_cm_dt)
        ax_cm_dt.set_xlabel('Предсказанные классы')
        ax_cm_dt.set_ylabel('Истинные классы')
        ax_cm_dt.set_title('Confusion Matrix')
        st.pyplot(fig_cm_dt)
        st.text("Classification Report:\n" + classification_report(st.session_state['y_test'], st.session_state['y_pred_dt']))

    # ---- Decision Boundary Plots (Conditional) ----
    if show_decision_boundaries and len(selected_features) == 2:
        st.header("Границы решений (для выбранных 2 признаков)")
        X_train_top2_np = st.session_state['X_train_selected'].values
        y_train_np = st.session_state['y_train'].values

        db_col1, db_col2, db_col3 = st.columns(3)

        with db_col1:
            st.subheader("KNN")
            knn_classifier_vis = KNeighborsClassifier(n_neighbors=n_neighbors_knn)
            knn_classifier_vis.fit(X_train_top2_np, y_train_np)
            fig_knn_db = plt.figure(figsize=(6, 5)) # Smaller figures for column layout
            plot_decision_regions(X_train_top2_np, y_train_np, clf=knn_classifier_vis, legend=2)
            plt.xlabel(selected_features[0].capitalize())
            plt.ylabel(selected_features[1].capitalize())
            plt.title('KNN Decision Boundary')
            plt.grid(True)
            st.pyplot(fig_knn_db)

        with db_col2:
            st.subheader("Logistic Regression")
            logistic_regression_classifier_vis = LogisticRegression(max_iter=565, random_state=42)
            logistic_regression_classifier_vis.fit(X_train_top2_np, y_train_np)
            fig_logreg_db = plt.figure(figsize=(6, 5))
            plot_decision_regions(X_train_top2_np, y_train_np, clf=logistic_regression_classifier_vis, legend=2)
            plt.xlabel(selected_features[0].capitalize())
            plt.ylabel(selected_features[1].capitalize())
            plt.title('Logistic Regression Decision Boundary')
            plt.grid(True)
            st.pyplot(fig_logreg_db)

        with db_col3:
            st.subheader("Decision Tree")
            decision_tree_classifier_vis = DecisionTreeClassifier(max_depth=max_depth_dt, random_state=42)
            decision_tree_classifier_vis.fit(X_train_top2_np, y_train_np)
            fig_dt_db = plt.figure(figsize=(6, 5))
            plot_decision_regions(X_train_top2_np, y_train_np, clf=decision_tree_classifier_vis, legend=2)
            plt.xlabel(selected_features[0].capitalize())
            plt.ylabel(selected_features[1].capitalize())
            plt.title('Decision Tree Decision Boundary')
            plt.grid(True)
            st.pyplot(fig_dt_db)
    elif show_decision_boundaries and len(selected_features) != 2:
        st.info("Границы решений отображаются только для 2 выбранных признаков. Выберите ровно 2 признака в боковой панели, чтобы увидеть их.")

    # ---- ROC Curves ----
    st.header("ROC-кривые")
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    ax_roc.plot(roc_curve(st.session_state['y_test'], st.session_state['y_prob_knn'])[0], roc_curve(st.session_state['y_test'], st.session_state['y_prob_knn'])[1], label=f'KNN (AUC = {roc_auc_score(st.session_state["y_test"], st.session_state["y_prob_knn"]):.2f})')
    ax_roc.plot(roc_curve(st.session_state['y_test'], st.session_state['y_prob_logreg'])[0], roc_curve(st.session_state['y_test'], st.session_state['y_prob_logreg'])[1], label=f'Logistic Regression (AUC = {roc_auc_score(st.session_state["y_test"], st.session_state["y_prob_logreg"]):.2f})')
    ax_roc.plot(roc_curve(st.session_state['y_test'], st.session_state['y_prob_dt'])[0], roc_curve(st.session_state['y_test'], st.session_state['y_prob_dt'])[1], label=f'Decision Tree (AUC = {roc_auc_score(st.session_state["y_test"], st.session_state["y_prob_dt"]):.2f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC-кривые для разных классификаторов')
    ax_roc.legend(title='Классификатор')
    ax_roc.grid(True)
    st.pyplot(fig_roc)

    # ---- Feature Importance (Decision Tree) ----
    if isinstance(st.session_state['dt_classifier'], DecisionTreeClassifier) and len(selected_features) >= 1:
        feature_importance = pd.DataFrame({'Feature': st.session_state['X_train_selected'].columns, 'Importance': st.session_state['dt_classifier'].feature_importances_})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)

        st.header("Важность признаков (Decision Tree)")
        fig_feature_importance, ax_feature_importance = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax_feature_importance)
        ax_feature_importance.set_xlabel('Важность')
        ax_feature_importance.set_ylabel('Признак')
        ax_feature_importance.set_title('Важность признаков для Decision Tree')
        st.pyplot(fig_feature_importance)

    # ---- AUC on Train/Test ----
    st.header("AUC на обучающей и тестовой выборках")
    auc_col1, auc_col2, auc_col3 = st.columns(3)

    with auc_col1:
        st.subheader("KNN")
        auc_train_knn = auc(roc_curve(st.session_state['y_train'], st.session_state['knn_classifier'].predict_proba(st.session_state['X_train_selected'])[:, 1])[0], roc_curve(st.session_state['y_train'], st.session_state['knn_classifier'].predict_proba(st.session_state['X_train_selected'])[:, 1])[1])
        auc_test_knn = auc(roc_curve(st.session_state['y_test'], st.session_state['y_prob_knn'])[0], roc_curve(st.session_state['y_test'], st.session_state['y_prob_knn'])[1])
        st.metric("AUC на трейне", f"{auc_train_knn:.2f}")
        st.metric("AUC на тесте", f"{auc_test_knn:.2f}")

    with auc_col2:
        st.subheader("Logistic Regression")
        auc_train_logreg = auc(roc_curve(st.session_state['y_train'], st.session_state['logreg_classifier'].predict_proba(st.session_state['X_train_selected'])[:, 1])[0], roc_curve(st.session_state['y_train'], st.session_state['logreg_classifier'].predict_proba(st.session_state['X_train_selected'])[:, 1])[1])
        auc_test_logreg = auc(roc_curve(st.session_state['y_test'], st.session_state['y_prob_logreg'])[0], roc_curve(st.session_state['y_test'], st.session_state['y_prob_logreg'])[1])
        st.metric("AUC на трейне", f"{auc_train_logreg:.2f}")
        st.metric("AUC на тесте", f"{auc_test_logreg:.2f}")

    with auc_col3:
        st.subheader("Decision Tree")
        auc_train_dt = auc(roc_curve(st.session_state['y_train'], st.session_state['dt_classifier'].predict_proba(st.session_state['X_train_selected'])[:, 1])[0], roc_curve(st.session_state['y_train'], st.session_state['dt_classifier'].predict_proba(st.session_state['X_train_selected'])[:, 1])[1])
        auc_test_dt = auc(roc_curve(st.session_state['y_test'], st.session_state['y_prob_dt'])[0], roc_curve(st.session_state['y_test'], st.session_state['y_prob_dt'])[1])
        st.metric("AUC на трейне", f"{auc_train_dt:.2f}")
        st.metric("AUC на тесте", f"{auc_test_dt:.2f}")
else:
    st.info("Нажмите кнопку 'Переобучить модели' в боковой панели, чтобы запустить обучение и оценку моделей.")

st.markdown("---")
st.markdown("Приложение Streamlit для анализа Anneal Dataset.  "
            "Использованы библиотеки: Streamlit, Pandas, Scikit-learn, Matplotlib, Seaborn, Plotly, MLxtend.")
