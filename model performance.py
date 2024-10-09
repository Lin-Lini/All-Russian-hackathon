import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle
from catboost import Pool

# Загрузка тестовых данных
test_data = pd.read_csv('proizvod_test_data.csv')  # Укажите путь к вашему тестовому датасету
y_true = pd.read_csv('proizvod_test_true.csv')  # Укажите путь к вашему датасету с правильными метками

# Проверка, что количество строк в обоих файлах совпадает
if test_data.shape[0] != y_true.shape[0]:
    raise ValueError("Количество строк в тестовом наборе и наборе меток не совпадает.")

# Конкатенация данных без индекса (по умолчанию индекс сбрасывается)
test_data = pd.concat([test_data, y_true[['age_class', 'sex']]], axis=1)

# Проверка наличия столбца 'view_count'
if 'view_count' not in test_data.columns:
    raise ValueError("Столбец 'view_count' не найден в данных.")

# Загрузка обученных моделей
with open('model_age_catboost_new.pkl', 'rb') as f:
    model_age = pickle.load(f)

with open('model_sex_catboost_new.pkl', 'rb') as f:
    model_sex = pickle.load(f)

# Разделение пользователей на категории по количеству просмотренных видео
test_data['view_count_group'] = pd.cut(test_data['view_count'],
                                         bins=[0, 5, 10, 20, 50, np.inf],
                                         labels=['0-5', '6-10', '11-20', '21-50', '50+'])

# Приведение категориальных признаков к строковому типу и заполнение NaN значений
categorical_features = ['region', 'ua_device_type', 'ua_client_type', 'ua_os', 'ua_client_name',
                        'title', 'category', 'season', 'watch_time_category', 'rutube_video_id']

for feature in categorical_features:
    test_data[feature] = test_data[feature].astype(str)  # Приводим к строковому типу
    test_data[feature] = test_data[feature].fillna('')  # Заполняем пропуски пустыми строками

# Группировка данных по количеству просмотренных видео
video_groups = test_data.groupby('view_count_group')

# Словарь для хранения метрик по каждой группе
metrics = {
    'group': [],
    'f1_age': [],
    'precision_age': [],
    'recall_age': [],
    'f1_sex': [],
    'precision_sex': [],
    'recall_sex': []
}

# Анализ каждой группы
for group_name, group_data in video_groups:
    if group_data.shape[0] > 0:
        # Преобразование 'watch_time_category' в числовой формат (если необходимо)
        group_data['watch_time_category'] = group_data['watch_time_category'].map({
            'short_watch': 0,
            'medium_watch': 1,
            'long_watch': 2
        }).fillna(-1)  # Заполняем NaN значением -1 или другим подходящим значением

        # Создаем Pool для возрастных прогнозов
        age_pool = Pool(
            data=group_data.drop(['age_class', 'sex', 'view_count_group'], axis=1),
            cat_features=categorical_features  # Указываем категориальные признаки
        )

        # Создаем Pool для половых прогнозов
        sex_pool = Pool(
            data=group_data.drop(['age_class', 'sex', 'view_count_group'], axis=1),
            cat_features=categorical_features  # Указываем категориальные признаки
        )

        # Прогнозы для текущей группы
        age_predictions = model_age.predict(age_pool)
        sex_predictions = model_sex.predict(sex_pool)

        # Целевые метки для текущей группы
        true_age = group_data['age_class']
        true_sex = group_data['sex']

        # Расчет метрик для возраста
        f1_age = f1_score(true_age, age_predictions, average='weighted')
        precision_age = precision_score(true_age, age_predictions, average='weighted')
        recall_age = recall_score(true_age, age_predictions, average='weighted')

        # Расчет метрик для пола
        f1_sex = f1_score(true_sex, sex_predictions, average='weighted')
        precision_sex = precision_score(true_sex, sex_predictions, average='weighted')
        recall_sex = recall_score(true_sex, sex_predictions, average='weighted')

        # Сохранение результатов
        metrics['group'].append(group_name)
        metrics['f1_age'].append(f1_age)
        metrics['precision_age'].append(precision_age)
        metrics['recall_age'].append(recall_age)
        metrics['f1_sex'].append(f1_sex)
        metrics['precision_sex'].append(precision_sex)
        metrics['recall_sex'].append(recall_sex)

# Преобразование результатов в DataFrame
metrics_df = pd.DataFrame(metrics)

# Создаем новую колонку с взвешенными метриками
metrics_df['weighted_f1'] = metrics_df['f1_sex'] * 0.7 + metrics_df['f1_age'] * 0.3

# Выводим обновленный DataFrame с новой колонкой
print(metrics_df)

# Визуализация взвешенных метрик
plt.figure(figsize=(12, 6))
plt.plot(metrics_df['group'], metrics_df['weighted_f1'], label='Weighted F1 Score', marker='o')
plt.xlabel('Количество просмотренных видео')
plt.ylabel('Weighted F1 Score')
plt.title('Weighted F1 Score от количества просмотренных видео')
plt.legend()
plt.grid(True)
plt.show()

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(metrics_df['group'], metrics_df['f1_age'], label='F1 Score (Возраст)', marker='o')
plt.plot(metrics_df['group'], metrics_df['f1_sex'], label='F1 Score (Пол)', marker='o')
plt.xlabel('Количество просмотренных видео')
plt.ylabel('F1 Score')
plt.title('Зависимость F1 Score от количества просмотренных видео')
plt.legend()
plt.grid(True)
plt.show()

# Сохранение метрик в файл
metrics_df.to_csv('video_count_metrics_test.csv', index=False)

# Определение минимального количества видео для хороших показателей
good_performance_threshold = 0.6  # Установите желаемый порог
min_video_count = metrics_df[metrics_df['weighted_f1'] >= good_performance_threshold]['group'].min()

print(f"Минимальное количество просмотренных видео для хороших показателей: {min_video_count}")
