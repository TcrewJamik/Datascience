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
    st.header("Выбор модели и гиперпараметров")
    model_choice = st.selectbox("Выберите модель:", ["KNN", "Logistic Regression", "Decision Tree"])

    if model_choice == "KNN":
        n_neighbors_knn = st.slider("n_neighbors", min_value=1, max_value=20, value=3, step=1)
    elif model_choice == "Decision Tree":
        max_depth_dt = st.slider("max_depth", min_value=1, max_value=10, value=5, step=1)

    st.markdown("---")
    st.header("Анализ признаков (Пример)")
    st.markdown("**Примечание:** Эти признаки ('Variance', 'Skewness', 'Curtosis', 'Entropy') являются примерами "
                "и не входят в исходный набор данных Anneal. "
                "Слайдеры и графики ниже демонстрируют, как можно анализировать признаки, если бы они были доступны.")

    variance_range = st.slider("Variance", min_value=-7.04, max_value=6.82, value=(-0.43, 0.43), step=0.01)
    skewness_range = st.slider("Skewness", min_value=-13.77, max_value=12.95, value=(-1.92, 1.92), step=0.01)
    curtosis_range = st.slider("Curtosis", min_value=-5.29, max_value=17.93, value=(-1.40, 1.40), step=0.01)
    entropy_range = st.slider("Entropy", min_value=-8.55, max_value=2.45, value=(-1.19, -1.19), step=0.01)

    st.markdown("---")
    available_features = X_train.columns.tolist()
    default_features = ['formability', 'condition'] if all(f in available_features for f in ['formability', 'condition']) else available_features[:min(2, len(available_features))]
    selected_features = st.multiselect("Выберите признаки для обучения моделей", available_features, default=default_features)
    show_decision_boundaries = st.checkbox("Показать границы решений", value=True)
    retrain_button = st.button("Переобучить модели")

# ---- Data Exploration Section ----
st.header("Обзор данных")
st.subheader("Исходные данные")
st.dataframe(data_original.head())

st.subheader("Описательная статистика")
st.dataframe(data.describe())

st.subheader("Распределение классов (бинарная целевая переменная)")
binary_class_counts = data["binary_class"].value_counts()
st.bar_chart(binary_class_counts)

# Feature Histograms
st.subheader("Гистограммы признаков")
feature_hist_cols = st.columns(3)
for i, col in enumerate(X_train.columns):
    with feature_hist_cols[i % 3]:
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(data=X_train, x=col, kde=True, ax=ax_hist)
        st.pyplot(fig_hist)

# ---- Example Feature Analysis Section ----
st.header("Анализ данных (Пример признаков)")
st.markdown("Визуализация распределения примерных признаков 'Variance', 'Skewness', 'Curtosis', 'Entropy'. "
            "Эти графики показывают, как могли бы выглядеть гистограммы и плотности распределения признаков.")

example_features = {
    "variance": np.random.normal(loc=0, scale=2, size=1000),
    "skewness": np.random.normal(loc=-2, scale=5, size=1000),
    "curtosis": np.random.normal(loc=3, scale=4, size=1000),
    "entropy": np.random.normal(loc=-1, scale=1.5, size=1000),
    "binary_class": np.random.randint(0, 2, size=1000) # Dummy binary class for density plots
}
example_df = pd.DataFrame(example_features)

example_feature_list = ["variance", "skewness", "curtosis", "entropy"]
example_plots_cols = st.columns(2) # 2 columns for plots

for i, feature_name in enumerate(example_feature_list):
    with example_plots_cols[i % 2]:
        st.subheader(f"Гистограмма: {feature_name}")
        fig_hist_example, ax_hist_example = plt.subplots()
        sns.histplot(example_df[feature_name], kde=True, ax=ax_hist_example)
        st.pyplot(fig_hist_example)

        st.subheader(f"Плотность: {feature_name} по классам")
        fig_density_example, ax_density_example = plt.subplots()
        sns.kdeplot(data=example_df, x=feature_name, hue="binary_class", fill=True, ax=ax_density_example)
        ax_density_example.legend(title='class', labels=['0', '1']) # Fix legend
        st.pyplot(fig_density_example)


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
if retrain_button or not st.session_state.get('models_trained', False):
    st.session_state['models_trained'] = True

    if len(selected_features) < 2:
        st.warning("Для обучения моделей и визуализации границ решений выберите как минимум два признака в боковой панели. Модели обучены на дефолтных признаках.")
        X_train_selected = X_train[default_features[:min(2, len(default_features))]]
        X_test_selected = X_test[default_features[:min(2, len(default_features))]]
    else:
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]


    # Model Training based on selected model
    if model_choice == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors_knn)
    elif model_choice == "Logistic Regression":
        classifier = LogisticRegression(max_iter=565, random_state=42, class_weight='balanced')
    elif model_choice == "Decision Tree":
        classifier = DecisionTreeClassifier(max_depth=max_depth_dt, random_state=42)
    else:
        classifier = LogisticRegression() # Default model

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
    st.session_state['model_choice'] = model_choice # Store model choice


# ---- Model Evaluation Display ----
st.header("Оценка моделей машинного обучения")

if st.session_state.get('models_trained', False):
    st.subheader(f"Оценка модели: {st.session_state['model_choice']}") # Dynamic model name
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{accuracy_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")
        st.metric("ROC AUC", f"{roc_auc_score(st.session_state['y_test'], st.session_state['y_prob']):.3f}")
        st.metric("F1", f"{f1_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")

        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Предсказанные классы')
        ax_cm.set_ylabel('Истинные классы')
        ax_cm.set_title('Confusion Matrix')
        st.pyplot(fig_cm)

    with col2:
        st.text("Classification Report:\n" + classification_report(st.session_state['y_test'], st.session_state['y_pred']))

        # ROC Curve
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(roc_curve(st.session_state['y_test'], st.session_state['y_prob'])[0], roc_curve(st.session_state['y_test'], st.session_state['y_prob'])[1],
                    label=f'{st.session_state["model_choice"]} (AUC = {roc_auc_score(st.session_state["y_test"], st.session_state["y_prob"]):.2f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC-кривая')
        ax_roc.legend(title='Классификатор')
        ax_roc.grid(True)
        st.pyplot(fig_roc)

    # ---- Decision Boundary Plots (Conditional) ----
    if show_decision_boundaries and len(selected_features) == 2:
        st.header("Границы решений (для выбранных 2 признаков)")
        X_train_top2_np = st.session_state['X_train_selected'].values
        y_train_np = st.session_state['y_train'].values

        st.subheader(st.session_state['model_choice'])
        classifier_vis = None
        if st.session_state['model_choice'] == "KNN":
            classifier_vis = KNeighborsClassifier(n_neighbors=n_neighbors_knn)
        elif st.session_state['model_choice'] == "Logistic Regression":
            classifier_vis = LogisticRegression(max_iter=565, random_state=42)
        elif st.session_state['model_choice'] == "Decision Tree":
            classifier_vis = DecisionTreeClassifier(max_depth=max_depth_dt, random_state=42)

        if classifier_vis: # Ensure classifier_vis is not None
            classifier_vis.fit(X_train_top2_np, y_train_np)
            fig_db = plt.figure(figsize=(6, 5))
            plot_decision_regions(X_train_top2_np, y_train_np, clf=classifier_vis, legend=2)
            plt.xlabel(selected_features[0].capitalize())
            plt.ylabel(selected_features[1].capitalize())
            plt.title(f'{st.session_state["model_choice"]} Decision Boundary')
            plt.grid(True)
            st.pyplot(fig_db)


    elif show_decision_boundaries and len(selected_features) != 2:
        st.info("Границы решений отображаются только для 2 выбранных признаков. Выберите ровно 2 признака в боковой панели, чтобы увидеть их.")

    # ---- Feature Importance (Decision Tree - Conditional) ----
    if st.session_state['model_choice'] == "Decision Tree" and len(selected_features) >= 1 and isinstance(st.session_state['classifier'], DecisionTreeClassifier):
        feature_importance = pd.DataFrame({'Feature': st.session_state['X_train_selected'].columns, 'Importance': st.session_state['classifier'].feature_importances_})
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
    auc_train = auc(roc_curve(st.session_state['y_train'], st.session_state['classifier'].predict_proba(st.session_state['X_train_selected'])[:, 1])[0], roc_curve(st.session_state['y_train'], st.session_state['classifier'].predict_proba(st.session_state['X_train_selected'])[:, 1])[1])
    auc_test = auc(roc_curve(st.session_state['y_test'], st.session_state['y_prob'])[0], roc_curve(st.session_state['y_test'], st.session_state['y_prob'])[1])
    st.metric("AUC на трейне", f"{auc_train:.2f}")
    st.metric("AUC на тесте", f"{auc_test:.2f}")

else:
    st.info("Нажмите кнопку 'Переобучить модели' в боковой панели, чтобы запустить обучение и оценку моделей.")

st.markdown("---")
st.markdown("Приложение Streamlit для анализа Anneal Dataset.  "
            "Использованы библиотеки: Streamlit, Pandas, Scikit-learn, Matplotlib, Seaborn, Plotly, MLxtend.")
