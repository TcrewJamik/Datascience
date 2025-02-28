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

st.set_page_config(page_title="Anneal Steel Explorer Pro", page_icon="⚙️", layout="wide")

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
st.title("⚙️ Anneal Steel Explorer Pro 🚀")
st.markdown("Продвинутое приложение для исследования и классификации Anneal Steel Dataset. "
            "Настройте модели, исследуйте данные и получите глубокое понимание!")

# ---- Sidebar for Controls ----
with st.sidebar:
    st.header("🛠️ Настройки модели")
    model_choice = st.selectbox("Выберите модель:", ["KNN", "Logistic Regression", "Decision Tree"])

    hyperparams = {}
    if model_choice == "KNN":
        hyperparams['n_neighbors'] = st.slider("n_neighbors", min_value=1, max_value=20, value=3, step=1)
        hyperparams['weights'] = st.selectbox("weights", options=['uniform', 'distance'], index=0)
        hyperparams['algorithm'] = st.selectbox("algorithm", options=['auto', 'ball_tree', 'kd_tree', 'brute'], index=0)
        hyperparams['p'] = st.slider("p (Minkowski distance power)", min_value=1, max_value=5, value=2, step=1)

    elif model_choice == "Logistic Regression":
        hyperparams['C'] = st.slider("C (Regularization)", min_value=0.001, max_value=10.0, step=0.01, value=1.0, format="%.3f")
        hyperparams['penalty'] = st.selectbox("penalty", options=['l1', 'l2', 'elasticnet', 'none'], index=1)
        hyperparams['solver'] = st.selectbox("solver", options=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], index=1)
        if hyperparams['penalty'] == 'elasticnet':
            hyperparams['l1_ratio'] = st.slider("l1_ratio (Elastic-Net)", min_value=0.0, max_value=1.0, step=0.05, value=0.5)

    elif model_choice == "Decision Tree":
        hyperparams['criterion'] = st.selectbox("criterion", options=['gini', 'entropy'], index=0)
        hyperparams['max_depth'] = st.slider("max_depth", min_value=1, max_value=20, value=5, step=1)
        hyperparams['min_samples_split'] = st.slider("min_samples_split", min_value=2, max_value=20, value=2, step=1)
        hyperparams['min_samples_leaf'] = st.slider("min_samples_leaf", min_value=1, max_value=10, value=1, step=1)
        hyperparams['max_features'] = st.selectbox("max_features", options=['auto', 'sqrt', 'log2', None], index=3)

    st.markdown("---")
    st.header("📊 Выбор признаков")
    available_features = X_train.columns.tolist()
    default_features = ['formability', 'condition'] if all(f in available_features for f in ['formability', 'condition']) else available_features[:min(2, len(available_features))]
    selected_features = st.multiselect("Выберите признаки для обучения:", available_features, default=default_features)
    show_decision_boundaries = st.checkbox("Показать границы решений", value=False) # Default to False to avoid initial error
    grid_points_value = int(st.slider("Плотность сетки границ решений", min_value=20, max_value=150, value=75, step=25)) # Explicitly cast to int
    retrain_button = st.button("🔥 Переобучить модель")

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
        st.info("Пропущенные значения отсутствуют или уже обработаны.")

    if st.checkbox("Показать гистограммы признаков"):
        st.subheader("Гистограммы признаков")
        feature_hist_cols = st.columns(3)
        for i, col in enumerate(X_train.columns):
            with feature_hist_cols[i % 3]:
                fig_hist, ax_hist = plt.subplots()
                sns.histplot(data=X_train, x=col, kde=True, ax=ax_hist)
                ax_hist.set_title(col, fontsize=10)
                st.pyplot(fig_hist, use_container_width=True)


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

    # Model Training with Hyperparameter Tuning
    if model_choice == "KNN":
        classifier = KNeighborsClassifier(**hyperparams)
    elif model_choice == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', **hyperparams) # Increased max_iter
    elif model_choice == "Decision Tree":
        classifier = DecisionTreeClassifier(random_state=42, **hyperparams)
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
    st.session_state['hyperparams'] = hyperparams # Store hyperparameters


# ---- Model Evaluation Display ----
st.header("🏆 Оценка модели")
if st.session_state.get('models_trained', False): # Conditional check here!
    st.subheader(f"Модель: {st.session_state['model_choice']}")
    st.write(f"Гиперпараметры: {st.session_state['hyperparams']}") # Now safe to access hyperparams

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
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax_cm) # viridis for better contrast
        ax_cm.set_xlabel('Предсказанные классы')
        ax_cm.set_ylabel('Истинные классы')
        ax_cm.set_title('Матрица ошибок (Confusion Matrix)')
        st.pyplot(fig_cm)

        # ROC Curve (Plotly)
        fpr, tpr, thresholds = roc_curve(st.session_state['y_test'], st.session_state['y_prob'])
        roc_auc = auc(fpr, tpr)

        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC-кривая (AUC = {roc_auc:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
        )
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_traces(fillcolor='rgba(99, 255, 132, 0.6)') # Vibrant green fill
        st.plotly_chart(fig_roc)

    st.subheader("Отчет о классификации")
    st.text(classification_report(st.session_state['y_test'], st.session_state['y_pred']))

    # ---- Decision Boundary Plots (Conditional) ----
    if show_decision_boundaries and len(selected_features) == 2:
        st.header("🗺️ Граница решений")
        X_train_top2_np = st.session_state['X_train_selected'].values
        y_train_np = st.session_state['y_train'].values

        st.info("Граница решений визуализируется для первых двух выбранных признаков.")

        with st.spinner("Отрисовка границ решений..."):
            classifier_vis = None
            if st.session_state['model_choice'] == "KNN":
                classifier_vis = KNeighborsClassifier(**st.session_state['hyperparams'])
            elif st.session_state['model_choice'] == "Logistic Regression":
                classifier_vis = LogisticRegression(max_iter=1000, random_state=42, **st.session_state['hyperparams'])
            elif st.session_state['model_choice'] == "Decision Tree":
                classifier_vis = DecisionTreeClassifier(random_state=42, **st.session_state['hyperparams'])

            if classifier_vis:
                print(f"Type of X_train_top2_np: {X_train_top2_np.dtype}, Shape: {X_train_top2_np.shape}") # DEBUG
                print(f"Type of y_train_np: {y_train_np.dtype}, Shape: {y_train_np.shape}") # DEBUG
                print(f"Type of classifier_vis: {type(classifier_vis)}") # DEBUG
                print(f"Grid points value: {grid_points_value}, Type: {type(grid_points_value)}") # DEBUG

                fig_db = plt.figure(figsize=(8, 6))
                plot_decision_regions(X_train_top2_np, y_train_np, clf=classifier_vis, legend=2, grid_points=grid_points_value)
                plt.xlabel(selected_features[0].capitalize())
                plt.ylabel(selected_features[1].capitalize())
                plt.title(f'Граница решений для {st.session_state["model_choice"]}')
                plt.grid(False) # Removed grid for cleaner look
                st.pyplot(fig_db)
            else:
                st.error("Не удалось отобразить границу решений.") # Indicate failure if classifier_vis is None

    elif show_decision_boundaries and len(selected_features) != 2:
        st.info("Границы решений отображаются только для 2 выбранных признаков. Выберите ровно 2 признака в боковой панели, чтобы увидеть их.")
else:
    st.info("Нажмите кнопку 'Переобучить модель' в боковой панели, чтобы запустить обучение и оценку модели.")

st.markdown("---")
st.markdown("🚀 **Anneal Steel Explorer Pro** | Интерактивное исследование и классификация  |  "
            "Разработано с любовью к данным и Streamlit.")
