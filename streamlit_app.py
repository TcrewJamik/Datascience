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

st.set_page_config(page_title="Исследование Anneal Dataset", page_icon="🔥", layout="wide")

# ---- Загрузка данных ----
file_path = "anneal.data"

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, sep=",", header=None, na_values=["?"])
    data.columns = [
        "famiily", "product-type", "steel", "carbon", "hardness", "temper-rolling", "condition", "formability",
        "strength", "non-ageing", "surface-finish", "surface-quality", "enamelability", "bc", "bf", "bt", "bw/me",
        "bl", "m", "chrom", "phos", "cbond", "marvi", "exptl", "ferro", "corr", "blue/bright/varn/clean",
        "lustre", "jurofm", "s", "p", "shape", "thick", "width", "len", "oil", "bore", "packing", "class"
    ]
    return data

data_original = load_data(file_path)
data = data_original.copy()

# ---- Предобработка данных ----
columns_to_drop = [
    "famiily", "temper-rolling", "non-ageing", "surface-finish", "enamelability", "bc", "bf", "bt", "bl", "m",
    "chrom", "phos", "cbond", "marvi", "exptl", "ferro", "corr", "blue/bright/varn/clean", "lustre", "jurofm",
    "s", "p", "oil", "packing", "bw/me"
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
st.title("🔥 Anneal Steel Dataset Explorer 🔥")
st.markdown("Интерактивное приложение для исследования и классификации данных Anneal Steel. "
            "Выберите модель, настройте параметры, и изучите результаты!")

# ---- Sidebar for Controls ----
with st.sidebar:
    st.header("⚙️ Настройки модели")
    model_choice = st.selectbox("Выберите модель:", ["KNN", "Logistic Regression", "Decision Tree"])

    hyperparams = {}
    if model_choice == "KNN":
        hyperparams['n_neighbors'] = st.slider("n_neighbors", min_value=1, max_value=20, value=3, step=1)
    elif model_choice == "Decision Tree":
        hyperparams['max_depth'] = st.slider("max_depth", min_value=1, max_value=10, value=5, step=1)

    st.markdown("---")
    st.header("📊 Выбор признаков")
    available_features = X_train.columns.tolist()
    default_features = ['formability', 'condition'] if all(f in available_features for f in ['formability', 'condition']) else available_features[:min(2, len(available_features))]
    selected_features = st.multiselect("Выберите признаки для обучения:", available_features, default=default_features)
    show_decision_boundaries = st.checkbox("Показать границы решений", value=True)
    retrain_button = st.button("🚀 Переобучить модель")

# ---- Data Exploration Section ----
expander_data_explore = st.expander("🔍 Исследование данных", expanded=False)
with expander_data_explore:
    st.subheader("Предварительный просмотр данных")
    st.dataframe(data_original.head())

    st.subheader("Описательная статистика")
    st.dataframe(data.describe())

    st.subheader("Распределение классов")
    binary_class_counts = data["binary_class"].value_counts()
    fig_class_dist = px.bar(binary_class_counts, x=binary_class_counts.index, y=binary_class_counts.values,
                             labels={'x': 'Класс', 'y': 'Количество'}, title="Распределение классов (Бинарная)")
    st.plotly_chart(fig_class_dist)

    st.subheader("Процент пропущенных значений (до обработки)")
    missing_percentage = data_original.isna().sum() / len(data_original) * 100
    missing_df = pd.DataFrame({'Признак': missing_percentage.index, 'Процент пропусков': missing_percentage.values})
    missing_df = missing_df[missing_df['Процент пропусков'] > 0].sort_values(by='Процент пропусков', ascending=False)

    if not missing_df.empty:
        fig_missing = px.bar(missing_df, x='Признак', y='Процент пропусков',
                                labels={'Процент пропусков': '% пропусков'},
                                title="Процент пропущенных значений в признаках")
        st.plotly_chart(fig_missing)
    else:
        st.info("Пропущенные значения отсутствуют в исходных данных или уже обработаны.")

    if st.checkbox("Показать гистограммы признаков"):
        st.subheader("Гистограммы признаков")
        feature_hist_cols = st.columns(3)
        for i, col in enumerate(X_train.columns):
            with feature_hist_cols[i % 3]:
                fig_hist, ax_hist = plt.subplots()
                sns.histplot(data=X_train, x=col, kde=True, ax=ax_hist)
                ax_hist.set_title(col, fontsize=10) # Smaller title for better layout
                st.pyplot(fig_hist, use_container_width=True) # use_container_width for better responsiveness


# ---- Model Training and Evaluation ----
if retrain_button or not st.session_state.get('models_trained', False):
    st.session_state['models_trained'] = True

    if len(selected_features) < 2:
        st.warning("Выберите как минимум два признака для обучения моделей и визуализации границ решений. "
                    "Модели обучены на дефолтных признаках.")
        X_train_selected = X_train[default_features[:min(2, len(default_features))]]
        X_test_selected = X_test[default_features[:min(2, len(default_features))]]
    else:
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

    # Model Training based on selected model and hyperparameters
    if model_choice == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=hyperparams.get('n_neighbors', 3))
    elif model_choice == "Logistic Regression":
        classifier = LogisticRegression(max_iter=565, random_state=42, class_weight='balanced')
    elif model_choice == "Decision Tree":
        classifier = DecisionTreeClassifier(max_depth=hyperparams.get('max_depth', 5), random_state=42)
    else:
        classifier = LogisticRegression()

    classifier.fit(X_train_selected, y_train)
    y_pred = classifier.predict(X_test_selected)
    y_prob = classifier.predict_proba(X_test_selected)[:, 1]

    st.session_state['classifier'] = classifier
    st.session_state['X_train_selected'] = X_train_selected
    st.session_state['y_train'] = y_train
    st.session_state['X_test_selected'] = X_test_selected
    st.session_state['y_test'] = y_test
    st.session_state['y_pred'] = y_pred
    st.session_state['y_prob'] = y_prob
    st.session_state['model_choice'] = model_choice

# ---- Model Evaluation Display ----
st.header("🏆 Оценка модели")
if st.session_state.get('models_trained', False):
    st.subheader(f"Модель: {st.session_state['model_choice']}")

    col_metrics, col_charts = st.columns(2)
    with col_metrics:
        st.metric("Точность (Accuracy)", f"{accuracy_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")
        st.metric("ROC AUC", f"{roc_auc_score(st.session_state['y_test'], st.session_state['y_prob']):.3f}")
        st.metric("F1-мера (F1 Score)", f"{f1_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")
        st.metric("Точность (Precision)", f"{precision_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")
        st.metric("Полнота (Recall)", f"{recall_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")

    with col_charts:
        # Confusion Matrix
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Предсказанные классы')
        ax_cm.set_ylabel('Истинные классы')
        ax_cm.set_title('Матрица ошибок (Confusion Matrix)')
        st.pyplot(fig_cm)

        # ROC Curve (Plotly for interactivity)
        fpr, tpr, thresholds = roc_curve(st.session_state['y_test'], st.session_state['y_prob'])
        roc_auc = auc(fpr, tpr)

        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC-кривая (AUC = {roc_auc:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
        )
        fig_roc.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig_roc.update_traces(fillcolor='rgba(4, 125, 224, 0.6)') # Slightly transparent fill
        st.plotly_chart(fig_roc)

    st.subheader("Отчет о классификации")
    st.text(classification_report(st.session_state['y_test'], st.session_state['y_pred']))

    # ---- Decision Boundary Plots (Conditional) ----
    if show_decision_boundaries and len(selected_features) == 2:
        st.header("🗺️ Граница решений")
        X_train_top2_np = st.session_state['X_train_selected'].values
        y_train_np = st.session_state['y_train'].values

        st.info("Граница решений визуализируется для первых двух выбранных признаков.")

        classifier_vis = None
        if st.session_state['model_choice'] == "KNN":
            classifier_vis = KNeighborsClassifier(n_neighbors=hyperparams.get('n_neighbors', 3))
        elif st.session_state['model_choice'] == "Logistic Regression":
            classifier_vis = LogisticRegression(max_iter=565, random_state=42)
        elif st.session_state['model_choice'] == "Decision Tree":
            classifier_vis = DecisionTreeClassifier(max_depth=hyperparams.get('max_depth', 5), random_state=42)

        if classifier_vis:
            classifier_vis.fit(X_train_top2_np, y_train_np)
            fig_db = plt.figure(figsize=(8, 6)) # Larger figure for decision boundary plot
            plot_decision_regions(X_train_top2_np, y_train_np, clf=classifier_vis, legend=2)
            plt.xlabel(selected_features[0].capitalize())
            plt.ylabel(selected_features[1].capitalize())
            plt.title(f'Граница решений для {st.session_state["model_choice"]}')
            plt.grid(True)
            st.pyplot(fig_db)

    elif show_decision_boundaries and len(selected_features) != 2:
        st.info("Границы решений отображаются только для 2 выбранных признаков. Выберите ровно 2 признака в боковой панели, чтобы увидеть их.")

    # ---- Feature Importance (Decision Tree - Conditional) ----
    if st.session_state['model_choice'] == "Decision Tree" and len(selected_features) >= 1 and isinstance(st.session_state['classifier'], DecisionTreeClassifier):
        feature_importance = pd.DataFrame({'Feature': st.session_state['X_train_selected'].columns, 'Importance': st.session_state['classifier'].feature_importances_})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)

        st.header("✨ Важность признаков")
        fig_feature_importance_plotly = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                                                labels={'Importance': 'Важность', 'Feature': 'Признак'},
                                                title='Важность признаков (Decision Tree)')
        st.plotly_chart(fig_feature_importance_plotly)

    # ---- AUC on Train/Test ----
    st.header("📊 AUC на обучающей и тестовой выборках")
    auc_train = auc(roc_curve(st.session_state['y_train'], st.session_state['classifier'].predict_proba(st.session_state['X_train_selected'])[:, 1])[0],
                    roc_curve(st.session_state['y_train'], st.session_state['classifier'].predict_proba(st.session_state['X_train_selected'])[:, 1])[1])
    auc_test = auc(roc_curve(st.session_state['y_test'], st.session_state['y_prob'])[0], roc_curve(st.session_state['y_test'], st.session_state['y_prob'])[1])
    st.metric("AUC на обучающей выборке", f"{auc_train:.2f}")
    st.metric("AUC на тестовой выборке", f"{auc_test:.2f}")


else:
    st.info("Нажмите кнопку 'Переобучить модель' в боковой панели, чтобы запустить обучение и оценку модели.")

st.markdown("---")
st.markdown("🚀 **Инструмент исследования Anneal Dataset** |  "
            "Разработано с использованием Streamlit, Pandas, Scikit-learn, Matplotlib, Seaborn, Plotly, MLxtend.")
