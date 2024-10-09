import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

train_data_cleaned = pd.read_csv('train_data_cleaned.csv')

# Определение категориальных признаков
categorical_features = [
    'region',
    'ua_device_type',
    'ua_client_type',
    'ua_os',
    'ua_client_name',
    'season',
    'watch_time_category',
    'category',
    'title',
    'rutube_video_id'
]

# Разделение на целевые переменные и признаки
X = train_data_cleaned.drop(['viewer_uid', 'age', 'sex'], axis=1)
y_age_class = train_data_cleaned['age_class'].astype(int)
y_sex = train_data_cleaned['sex'].astype(str)

# Разделение на обучающую и валидационную выборки
TRAIN_IDS, VAL_IDS = train_test_split(train_data_cleaned['viewer_uid'].unique(), train_size=0.8, shuffle=True, random_state=11)
print("Данные разделены на обучающую и валидационную выборки.")

# Обучающая выборка
train_events = train_data_cleaned[train_data_cleaned['viewer_uid'].isin(TRAIN_IDS)]
train_X = train_events.drop(['viewer_uid', 'age', 'sex', 'age_class'], axis=1)

# Валидационная выборка
val_events = train_data_cleaned[train_data_cleaned['viewer_uid'].isin(VAL_IDS)]
val_X = val_events.drop(['viewer_uid', 'age', 'sex', 'age_class'], axis=1)

# Сохранение данных для предсказаний (тестовая выборка)
test_data = val_events.drop(['age', 'sex', 'age_class'], axis=1)  # Удаляем целевые переменные
test_data.to_csv('proizvod_test_data.csv', index=False)  # Сохраняем тестовые данные

# Сохранение правильных ответов для валидационной выборки
val_labels = val_events[['age', 'sex', 'age_class']]
val_labels.to_csv('proizvod_test_true.csv', index=False)  # Сохраняем правильные ответы

# Приведение типов категориальных признаков к строкам
for col in categorical_features:
    train_X[col] = train_X[col].astype(str)
    val_X[col] = val_X[col].astype(str)

# Преобразуем в числовой формат только некатегориальные столбцы, чтобы избежать лишнего использования памяти
non_categorical_columns = [col for col in train_X.columns if col not in categorical_features]

# Применяем pd.to_numeric только к некатегориальным столбцам и заполняем NaN значениями 0
train_X[non_categorical_columns] = train_X[non_categorical_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
val_X[non_categorical_columns] = val_X[non_categorical_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# Оптимизация использования памяти: преобразуем числовые столбцы в float32, чтобы уменьшить объем используемой памяти
train_X[non_categorical_columns] = train_X[non_categorical_columns].astype('float32')
val_X[non_categorical_columns] = val_X[non_categorical_columns].astype('float32')

# Категориальные столбцы оставляем строками и заменяем 'nan' на 'missing'
for col in categorical_features:
    train_X[col] = train_X[col].astype(str).replace('nan', 'missing')
    val_X[col] = val_X[col].astype(str).replace('nan', 'missing')

# Создание Pool для обучения
try:
    train_pool_age = Pool(data=train_X, label=y_age_class[train_events.index], cat_features=categorical_features)
    train_pool_sex = Pool(data=train_X, label=y_sex[train_events.index], cat_features=categorical_features)
except Exception as e:
    print(f"Ошибка при создании пула: {e}")

# Создание и обучение модели для предсказания возраста
model_age = None
if 'train_pool_age' in locals():
    model_age = CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        bagging_temperature=0.2,
        border_count=64,
        thread_count=-1,
        verbose=0
    )

    # Обучение модели
    model_age.fit(train_pool_age)

# Создание Pool для пола
if 'train_pool_sex' in locals():
    val_pool_age = Pool(data=val_X, label=val_events['age_class'], cat_features=categorical_features)
    val_pool_sex = Pool(data=val_X, label=val_events['sex'], cat_features=categorical_features)

    # Создание и обучение модели для предсказания пола
    model_sex = CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        bagging_temperature=0.2,
        border_count=64,
        thread_count=-1,
        verbose=0
    )
    model_sex.fit(train_pool_sex)

# Сохранение моделей
try:
    if model_age:
        pickle.dump(model_age, open('model_age_catboost_new.pkl', 'wb'))
    if model_sex:
        pickle.dump(model_sex, open('model_sex_catboost_new.pkl', 'wb'))
except Exception as e:
    print(f"Ошибка при сохранении моделей: {e}")

# Оценка производительности моделей

# Убедитесь, что модели определены перед вызовом
# Проверяем, обучены ли модели
if 'model_age' in locals() and 'model_sex' in locals():

    # Предсказания для всей валидационной выборки
    age_predictions = model_age.predict(val_X)
    sex_predictions = model_sex.predict(val_X)
    viewer_ids = val_events['viewer_uid'].values

    # Проверка формы массивов
    print("Shape of age_predictions:", np.shape(age_predictions))
    print("Shape of sex_predictions:", np.shape(sex_predictions))
    print("Shape of viewer_ids:", np.shape(viewer_ids))

    # Преобразуем массивы в одномерные, если это необходимо
    age_predictions = np.array(age_predictions).flatten()
    sex_predictions = np.array(sex_predictions).flatten()
    viewer_ids = np.array(viewer_ids).flatten()

    # Создаем DataFrame с предсказаниями
    predictions_df = pd.DataFrame({
        'viewer_uid': viewer_ids,
        'predicted_age_class': age_predictions,
        'predicted_sex': sex_predictions
    })

    # Сохранение предсказаний в CSV файл
    predictions_df.to_csv('predictions_validation.csv', index=False)
    print("Предсказания успешно сохранены в файл 'predictions_validation.csv'.")

    # Оценка F1 Score для предсказаний
    f1_age = f1_score(val_events['age_class'], age_predictions, average='weighted')
    f1_sex = f1_score(val_events['sex'], sex_predictions, average='weighted')

    # Расчет общего взвешенного F1 Score
    weighted_f1 = 0.3 * f1_age + 0.7 * f1_sex

    # Визуализация метрик
    metrics = ['F1 Score (возраст)', 'F1 Score (пол)', 'Общий взвешенный F1']
    scores = [f1_age, f1_sex, weighted_f1]

    plt.figure(figsize=(8, 5))
    plt.bar(metrics, scores, color=['blue', 'orange', 'green'])
    plt.ylabel('Скор')
    plt.title('Метрики точности')
    plt.grid(axis='y')
    plt.show()

else:
    print("Модели не были обучены, проверьте наличие ошибок.")