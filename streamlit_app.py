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
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Anneal DataSet", page_icon="⚙️", layout="wide")

# ---- Загрузка данных ----
file_path = "anneal.data"

@st.cache_data  # Кэшируем загрузку данных для скорости
def load_data(file_path):
    data = pd.read_csv(file_path, sep=",", header=None, na_values=["?"])
    data.columns = [
        "famiily", "product-type", "steel", "carbon", "hardness", "temper-rolling", "condition", "formability",
        "strength", "non-ageing", "surface-finish", "surface-quality", "enamelability", "bc", "bf", "bt", "bw/me",
        "bl", "m", "chrom", "phos", "cbond", "marvi", "exptl", "ferro", "corr", "blue/bright/varn/clean",
        "lustre", "jurofm", "s", "p", "shape", "thick", "width", "len", "oil", "bore", "packing", "class"
    ]
    return data

data_original = load_data(file_path) # Загружаем исходные данные
data = data_original.copy() # Создаем копию для обработки

# ---- Предобработка данных ----
# Список колонок для удаления (много пропусков или неинформативны)
columns_to_drop = [
    "famiily", "temper-rolling", "non-ageing", "surface-finish", "enamelability", "bc", "bf", "bt", "bl", "m",
    "chrom", "phos", "cbond", "marvi", "exptl", "ferro", "corr", "blue/bright/varn/clean", "lustre", "jurofm",
    "s", "p", "oil", "packing", "bw/me"
]
data.drop(columns=columns_to_drop, inplace=True) # Удаляем колонки
data.drop(columns=['carbon', 'hardness', 'strength', 'bore', 'product-type'], inplace=True) # Удаляем еще колонки
data.dropna(subset=["class"], inplace=True) # Удаляем строки с пропусками в 'class'

# Преобразуем целевую переменную в бинарную (две группы классов)
class_counts = data["class"].value_counts()
if len(data["class"].unique()) > 2:
    median_freq = class_counts.median()
    group1 = class_counts[class_counts >= median_freq].index.tolist()
    data["binary_class"] = data["class"].apply(lambda x: 1 if x in group1 else 0) # Группируем классы
else:
    data["binary_class"] = data["class"]
data.drop('class', axis=1, inplace=True) # Удаляем исходную колонку 'class'

# Заполняем пропуски в категориальных колонках модой
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_value = data[col].mode()[0] # Вычисляем моду
    data[col].fillna(mode_value, inplace=True) # Заполняем модой
median_formability = data['formability'].median() # Вычисляем медиану для 'formability'
data['formability'].fillna(median_formability, inplace=True) # Заполняем медианой

# Кодируем категориальные признаки числовыми значениями
label_encoder = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col]) # Кодируем

# ---- Разделение данных ----
X = data.drop('binary_class', axis=1) # Признаки - все колонки, кроме целевой
y = data['binary_class'] # Целевая переменная
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Разделяем на train и test

# Масштабирование признаков
scaler = StandardScaler()
numerical_cols = X_train.columns # Числовые колонки
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols]) # Масштабируем train
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols]) # Масштабируем test

# ---- Streamlit App Layout ----
st.title("⚙️ Anneal DataSet") # Заголовок приложения

# ---- Sidebar for Controls ----
with st.sidebar:
    st.header("🛠️ Настройки модели") # Заголовок боковой панели
    model_choice = st.selectbox("Выберите модель:", ["KNN", "Logistic Regression", "Decision Tree"]) # Выбор модели

    hyperparams = {} # Словарь для гиперпараметров
    if model_choice == "KNN": # Настройки для KNN
        hyperparams['n_neighbors'] = st.slider("n_neighbors", min_value=1, max_value=20, value=3, step=1)
        hyperparams['weights'] = st.selectbox("weights", options=['uniform', 'distance'], index=0)
        hyperparams['metric'] = st.selectbox("metric", options=['minkowski', 'euclidean', 'manhattan', 'chebyshev'], index=0)
        hyperparams['p'] = st.slider("p (Minkowski distance power)", min_value=1, max_value=5, value=2, step=1)

    elif model_choice == "Logistic Regression": # Настройки для Logistic Regression
        hyperparams['C'] = st.slider("C (Regularization)", min_value=0.001, max_value=10.0, step=0.01, value=1.0, format="%.3f")
        hyperparams['penalty'] = st.selectbox("penalty", options=['l1', 'l2', 'none'], index=1)
        hyperparams['solver'] = st.selectbox("solver", options=['lbfgs', 'liblinear'], index=1)

    elif model_choice == "Decision Tree": # Настройки для Decision Tree
        hyperparams['criterion'] = st.selectbox("criterion", options=['gini', 'entropy'], index=0)
        hyperparams['max_depth'] = st.slider("max_depth", min_value=1, max_value=20, value=5, step=1)
        hyperparams['min_samples_split'] = st.slider("min_samples_split", min_value=2, max_value=20, value=2, step=1)
        hyperparams['min_samples_leaf'] = st.slider("min_samples_leaf", min_value=1, max_value=10, value=1, step=1)
        hyperparams['max_features'] = st.selectbox("max_features", options=['auto', 'sqrt', 'log2', None], index=3)

    st.markdown("---")
    st.header("📊 Выбор признаков") # Заголовок выбора признаков
    available_features = X_train.columns.tolist() # Список доступных признаков
    default_features = ['formability', 'condition'] if all(f in available_features for f in ['formability', 'condition']) else available_features[:min(2, len(available_features))] # Признаки по умолчанию
    selected_features = st.multiselect("Выберите признаки для обучения:", available_features, default=default_features) # Выбор признаков
    retrain_button = st.button("🔥 Переобучить модель") # Кнопка "Переобучить"

# ---- Data Exploration Section ----
expander_data_explore = st.expander("🔍 Исследование данных", expanded=False) # Раздел "Исследование данных" в expander
with expander_data_explore:
    st.subheader("Предварительный просмотр данных") # Подзаголовок
    st.dataframe(data_original.head()) # Показываем первые строки данных

    st.subheader("Описательная статистика") # Подзаголовок
    st.dataframe(data.describe()) # Описательная статистика

    st.subheader("Распределение классов") # Подзаголовок
    binary_class_counts = data["binary_class"].value_counts() # Распределение классов
    fig_class_dist = px.bar(binary_class_counts, x=binary_class_counts.index, y=binary_class_counts.values,
                             labels={'x': 'Класс', 'y': 'Количество'}, title="Распределение классов (Бинарная)") # Bar chart Plotly
    st.plotly_chart(fig_class_dist) # Выводим график

    st.subheader("Процент пропущенных значений (до обработки)") # Подзаголовок
    missing_percentage = data_original.isna().sum() / len(data_original) * 100 # Процент пропусков
    missing_df = pd.DataFrame({'Признак': missing_percentage.index, 'Процент пропусков': missing_percentage.values}) # DataFrame для пропусков
    missing_df = missing_df[missing_df['Процент пропусков'] > 0].sort_values(by='Процент пропусков', ascending=False) # Фильтруем и сортируем

    if not missing_df.empty: # Если есть пропуски, показываем график
        fig_missing = px.bar(missing_df, x='Признак', y='Процент пропусков',
                                labels={'Процент пропусков': '% пропусков'},
                                title="Процент пропущенных значений в признаках") # Bar chart Plotly
        st.plotly_chart(fig_missing) # Выводим график
    else: # Иначе - сообщение об отсутствии пропусков
        st.info("Пропущенные значения отсутствуют или уже обработаны.")

    if st.checkbox("Показать гистограммы признаков"): # Чекбокс для гистограмм
        st.subheader("Гистограммы признаков") # Подзаголовок
        feature_hist_cols = st.columns(3) # Разделяем на 3 колонки
        for i, col in enumerate(X_train.columns): # Итерируемся по признакам
            with feature_hist_cols[i % 3]: # Выбираем колонку для графика
                fig_hist, ax_hist = plt.subplots() # Создаем figure и axes
                sns.histplot(data=X_train, x=col, kde=True, ax=ax_hist) # Гистограмма Seaborn
                ax_hist.set_title(col, fontsize=10) # Заголовок графика
                st.pyplot(fig_hist, use_container_width=True) # Выводим график


# ---- Model Training and Evaluation ----
if retrain_button or not st.session_state.get('models_trained', False): # Обучаем модель при нажатии кнопки или первом запуске
    st.session_state['models_trained'] = True # Флаг, что модели обучены

    if len(selected_features) < 2: # Предупреждение, если выбрано меньше 2 признаков
        st.warning("Выберите как минимум два признака для обучения моделей. Модели обучены на дефолтных признаках.")
        X_train_selected = X_train[default_features[:min(2, len(default_features))]] # Используем дефолтные признаки
        X_test_selected = X_test[default_features[:min(2, len(default_features))]]
    else:
        X_train_selected = X_train[selected_features] # Используем выбранные признаки
        X_test_selected = X_test[selected_features]

    # Выбор и обучение модели на основе настроек
    if model_choice == "KNN":
        classifier = KNeighborsClassifier(**hyperparams) # KNN с гиперпараметрами
    elif model_choice == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', **hyperparams) # Logistic Regression с гиперпараметрами
    elif model_choice == "Decision Tree":
        classifier = DecisionTreeClassifier(random_state=42, **hyperparams) # Decision Tree с гиперпараметрами
    else:
        classifier = LogisticRegression() # Default модель

    classifier.fit(X_train_selected, y_train) # Обучаем модель
    y_pred = classifier.predict(X_test_selected) # Предсказываем на тесте
    y_prob = classifier.predict_proba(X_test_selected)[:, 1] # Вероятности для ROC-AUC

    # Сохраняем обученную модель и данные в session_state
    st.session_state['classifier'] = classifier
    st.session_state['X_train_selected'] = X_train_selected
    st.session_state['y_train'] = y_train
    st.session_state['X_test_selected'] = X_test_selected
    st.session_state['y_test'] = y_test
    st.session_state['y_pred'] = y_pred
    st.session_state['y_prob'] = y_prob
    st.session_state['model_choice'] = model_choice
    st.session_state['hyperparams'] = hyperparams

# ---- Model Evaluation Display ----
st.header("🏆 Оценка модели") # Заголовок раздела оценки
if st.session_state.get('models_trained', False): # Показываем оценку, только если модель обучена
    st.subheader(f"Модель: {st.session_state['model_choice']}") # Подзаголовок с названием модели
    st.write(f"Гиперпараметры: {st.session_state['hyperparams']}") # Выводим гиперпараметры

    col_metrics, col_charts = st.columns(2) # Разделяем на 2 колонки для метрик и графиков
    with col_metrics:
        st.metric("Точность (Accuracy)", f"{accuracy_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}") # Метрика Accuracy
        st.metric("ROC AUC", f"{roc_auc_score(st.session_state['y_test'], st.session_state['y_prob']):.3f}") # Метрика ROC AUC
        st.metric("F1-мера (F1 Score)", f"{f1_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}") # Метрика F1-score

    with col_charts:
        # Confusion Matrix
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred']) # Матрица ошибок
        fig_cm, ax_cm = plt.subplots() # Создаем figure и axes
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax_cm) # Heatmap для матрицы ошибок
        ax_cm.set_xlabel('Предсказанные классы') # Подпись оси x
        ax_cm.set_ylabel('Истинные классы') # Подпись оси y
        ax_cm.set_title('Матрица ошибок (Confusion Matrix)') # Заголовок графика
        st.pyplot(fig_cm) # Выводим график

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(st.session_state['y_test'], st.session_state['y_prob']) # ROC кривая
        roc_auc = auc(fpr, tpr) # AUC значение

        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC-кривая (AUC = {roc_auc:.2f})', # Заголовок графика с AUC
            labels=dict(x='False Positive Rate', y='True Positive Rate'), # Подписи осей
        ) # Area chart Plotly для ROC-кривой
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1) # Диагональная линия для случайного классификатора
        fig_roc.update_traces(fillcolor='rgba(99, 255, 132, 0.6)') # Цвет заливки кривой
        st.plotly_chart(fig_roc) # Выводим график

    st.subheader("Отчет о классификации") # Подзаголовок
    st.text(classification_report(st.session_state['y_test'], st.session_state['y_pred'])) # Выводим отчет о классификации

    # ---- Feature Importance (Decision Tree - Conditional) ----
    if st.session_state['model_choice'] == "Decision Tree" and isinstance(st.session_state['classifier'], DecisionTreeClassifier) and len(selected_features) >= 1: # Показываем важность признаков только для Decision Tree
        feature_importance = pd.DataFrame({'Feature': st.session_state['X_train_selected'].columns, 'Importance': st.session_state['classifier'].feature_importances_}) # DataFrame для важности признаков
        feature_importance = feature_importance.sort_values('Importance', ascending=False) # Сортируем по важности

        st.header("✨ Важность признаков") # Заголовок раздела важности признаков
        fig_feature_importance_plotly = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                                                labels={'Importance': 'Важность', 'Feature': 'Признак'},
                                                title='Важность признаков (Decision Tree)') # Bar chart Plotly для важности признаков
        st.plotly_chart(fig_feature_importance_plotly) # Выводим график

    # ---- AUC on Train/Test ----
    st.header("📊 AUC на обучающей и тестовой выборках") # Заголовок раздела AUC
    auc_train = auc(roc_curve(st.session_state['y_train'], st.session_state['classifier'].predict_proba(st.session_state['X_train_selected'])[:, 1])[0],
                    roc_curve(st.session_state['y_train'], st.session_state['classifier'].predict_proba(st.session_state['X_train_selected'])[:, 1])[1]) # AUC на train
    auc_test = auc(roc_curve(st.session_state['y_test'], st.session_state['y_prob'])[0], roc_curve(st.session_state['y_test'], st.session_state['y_prob'])[1]) # AUC на test
    col_auc_train, col_auc_test = st.columns(2) # Разделяем на 2 колонки для AUC train и test
    col_auc_train.metric("AUC на обучающей выборке", f"{auc_train:.2f}") # Метрика AUC на train
    col_auc_test.metric("AUC на тестовой выборке", f"{auc_test:.2f}") # Метрика AUC на test


else: # Сообщение, если модель еще не обучена
    st.info("Нажмите кнопку 'Переобучить модель' в боковой панели, чтобы запустить обучение и оценку модели.")

st.markdown("---")
st.markdown("Разработано компанией Jamshed Corporation совместно с ZyplAI") # Подпись внизу
