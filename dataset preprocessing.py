import pandas as pd
import numpy as np
import pytz
import random
import joblib
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer

path = 'C:/Users/user/Desktop/Хаки/ВсеросИИ/Идеальное решение/'

train_events = pd.read_csv(path + 'train_events.csv')
video_info = pd.read_csv(path + 'video_info_v2.csv')
train_targets = pd.read_csv(path + 'train_targets.csv')


# Объединение данных
train_data_cleaned = train_events.merge(train_targets, on='viewer_uid')
train_data_cleaned = train_data_cleaned.merge(video_info, on='rutube_video_id', how='left')

# Удаление строк с пропусками
train_data_cleaned.dropna(inplace=True)
print("Удалены все строки с пропусками.")

# Функция для получения временной зоны на основе региона
def get_timezone(region):
    region_city_map = {
        'Moscow': 'Europe/Moscow',
        'Adygeya Republic': 'Europe/Moscow',
        'Amur Oblast': 'Asia/Yakutsk',
        'Altai': 'Asia/Barnaul',
        'Altay Kray': 'Asia/Barnaul',
        'Arkhangelskaya': 'Europe/Moscow',
        'Astrakhan': 'Europe/Astrakhan',
        'Astrakhan Oblast': 'Europe/Astrakhan',
        'Bashkortostan Republic': 'Asia/Yekaterinburg',
        'Belgorod Oblast': 'Europe/Moscow',
        'Bryansk Oblast': 'Europe/Moscow',
        'Buryatiya Republic': 'Asia/Irkutsk',
        'Chechnya': 'Europe/Moscow',
        'Chelyabinsk': 'Asia/Yekaterinburg',
        'Chukotka': 'Asia/Anadyr',
        'Chuvashia': 'Europe/Moscow',
        'Dagestan': 'Europe/Moscow',
        'Ingushetiya Republic': 'Europe/Moscow',
        'Irkutsk Oblast': 'Asia/Irkutsk',
        'Kaliningrad': 'Europe/Kaliningrad',
        'Kaliningrad Oblast': 'Europe/Kaliningrad',
        'Kaluga': 'Europe/Moscow',
        'Kaluga Oblast': 'Europe/Moscow',
        'Kamchatka': 'Asia/Kamchatka',
        'Karachayevo-Cherkesiya Republic': 'Europe/Moscow',
        'Karelia': 'Europe/Moscow',
        'Kemerovo Oblast': 'Asia/Novosibirsk',
        'Khakasiya Republic': 'Asia/Krasnoyarsk',
        'Khanty-Mansia': 'Asia/Yekaterinburg',
        'Kirov': 'Europe/Moscow',
        'Kirov Oblast': 'Europe/Moscow',
        'Krasnodar Krai': 'Europe/Moscow',
        'Khabarovsk': 'Asia/Vladivostok',
        'Krasnoyarsk Krai': 'Asia/Vladivostok',
        'Kursk': 'Europe/Moscow',
        'Kursk Oblast': 'Europe/Moscow',
        'Leningradskaya Oblast': 'Europe/Moscow',
        'Lipetsk Oblast': 'Europe/Moscow',
        'Magadan Oblast': 'Asia/Magadan',
        'Mariy-El Republic': 'Europe/Moscow',
        'Murmansk': 'Europe/Moscow',
        'Nizhny Novgorod Oblast': 'Europe/Moscow',
        'Nenets': 'Europe/Moscow',
        'Omsk': 'Asia/Omsk',
        'Omsk Oblast': 'Asia/Omsk',
        'Primorskiy (Maritime) Kray': 'Asia/Vladivostok',
        'Primorye': 'Asia/Vladivostok',
        'Penza': 'Europe/Moscow',
        'Penza Oblast': 'Europe/Moscow',
        'Perm': 'Asia/Yekaterinburg',
        'Perm Krai': 'Asia/Yekaterinburg',
        'Pskov Oblast': 'Europe/Moscow',
        'Ryazan Oblast': 'Europe/Moscow',
        'Rostov': 'Europe/Moscow',
        'Sakhalin Oblast': 'Asia/Sakhalin',
        'Sakha': 'Asia/Yakutsk',
        'Samara Oblast': 'Europe/Samara',
        'Saratov Oblast': 'Europe/Samara',
        'Saratovskaya Oblast': 'Europe/Samara',
        'Sebastopol City': 'Europe/Moscow',
        'Smolensk': 'Europe/Moscow',
        'Smolensk Oblast': 'Europe/Moscow',
        'Sverdlovsk': 'Asia/Yekaterinburg',
        'Sverdlovsk Oblast': 'Asia/Yekaterinburg',
        'St.-Petersburg': 'Europe/Moscow',
        'Stavropol Krai': 'Europe/Moscow',
        'Stavropol’ Kray': 'Europe/Moscow',
        'Tatarstan Republic': 'Europe/Moscow',
        'Tver Oblast': 'Europe/Moscow',
        'Tyva Republic': 'Asia/Krasnoyarsk',
        'Tyumen Oblast': 'Asia/Yekaterinburg',
        'Udmurtiya Republic': 'Asia/Yekaterinburg',
        'Ulyanovsk': 'Europe/Samara',
        'Voronezh Oblast': 'Europe/Moscow',
        'Vladimir': 'Europe/Moscow',
        'Vladimir Oblast': 'Europe/Moscow',
        'Vologda': 'Europe/Moscow',
        'Vologda Oblast': 'Europe/Moscow',
        'Volgograd Oblast': 'Europe/Moscow',
        'Yamalo-Nenets': 'Asia/Yekaterinburg',
        'Yaroslavl Oblast': 'Europe/Moscow',
        'Zabaykalskiy (Transbaikal) Kray': 'Asia/Irkutsk'
    }
    return region_city_map.get(region, 'Europe/Moscow')

# Преобразование временной метки в datetime с учетом ошибок
train_data_cleaned['event_timestamp'] = pd.to_datetime(train_data_cleaned['event_timestamp'], errors='coerce', utc=True)

# Применение функции для перевода времени в местное
def convert_to_local_time(row):
    timezone = get_timezone(row['region'])
    local_tz = pytz.timezone(timezone)
    try:
        utc_time = pd.to_datetime(row['event_timestamp'])
        local_time = utc_time.astimezone(local_tz)
        return local_time
    except Exception as e:
        print(f"Error converting time for row: {row}, error: {e}")
        return None

# Преобразование временной метки в местное время
train_data_cleaned['event_timestamp'] = pd.to_datetime(train_data_cleaned['event_timestamp'], errors='coerce', utc=True)

# Создание новых признаков
train_data_cleaned['hour'] = train_data_cleaned['event_timestamp'].dt.hour
train_data_cleaned['day_of_week'] = train_data_cleaned['event_timestamp'].dt.dayofweek
train_data_cleaned['month'] = train_data_cleaned['event_timestamp'].dt.month
train_data_cleaned['season'] = train_data_cleaned['month'].apply(lambda x: 'winter' if x in [12, 1, 2] else
                                                                 'spring' if x in [3, 4, 5] else
                                                                 'summer' if x in [6, 7, 8] else 'fall')

# Признаки длительности просмотра
train_data_cleaned['watch_time_category'] = pd.cut(train_data_cleaned['total_watchtime'],
                                                   bins=[0, 600, 1800, np.inf],
                                                   labels=['short_watch', 'medium_watch', 'long_watch'])

# Группировка по пользователям для создания признаков активности
user_activity = train_data_cleaned.groupby('viewer_uid').agg({
    'total_watchtime': 'sum',
    'rutube_video_id': 'count'
}).rename(columns={'rutube_video_id': 'view_count'}).reset_index()

# Переименуем столбец total_watchtime в user_activity
user_activity.rename(columns={'total_watchtime': 'total_watchtime_user_activity'}, inplace=True)

# Объединение признаков активности с основными данными
train_data_cleaned = train_data_cleaned.merge(user_activity, on='viewer_uid', how='left')

# Обогащение датасета
train_data_cleaned = train_data_cleaned.dropna()

# Агрегируем данные по всем необходимым столбцам
grouped_df = train_data_cleaned.groupby(['region', 'ua_device_type', 'ua_client_type', 'ua_os', 'ua_client_name',
                         'sex', 'category', 'duration', 'hour', 'day_of_week', 'month', 'season',
                         'age_class']).size().reset_index(name='count')

# Добавляем столбец с процентом для каждой группы
grouped_df['percentage'] = (grouped_df['count'] / grouped_df['count'].sum()) * 100

# Фильтруем по возрастным группам 0 и 3
group_0_df = grouped_df[grouped_df['age_class'] == 0]
group_3_df = grouped_df[grouped_df['age_class'] == 3]

# Оставляем только нужные столбцы: все исходные и процент
group_0_result = group_0_df[['region', 'ua_device_type', 'ua_client_type', 'ua_os', 'ua_client_name',
                             'sex', 'category', 'duration', 'hour', 'day_of_week', 'month', 'season', 'percentage']]

group_3_result = group_3_df[['region', 'ua_device_type', 'ua_client_type', 'ua_os', 'ua_client_name',
                             'sex', 'category', 'duration', 'hour', 'day_of_week', 'month', 'season', 'percentage']]

# Сохраняем результаты в разные CSV файлы
group_0_result.to_csv('age_group_0_percentage.csv', index=False)
group_3_result.to_csv('age_group_3_percentage.csv', index=False)

# Проверка результатов
print("Группа 0:")
print(group_0_result.head())
print("\nГруппа 3:")
print(group_3_result.head())

# Загрузка данных с вероятностями для возрастных категорий 0 и 3
probabilities_age_0_df = pd.read_csv('age_group_0_percentage.csv', encoding='UTF-8', low_memory=False)
probabilities_age_3_df = pd.read_csv('age_group_3_percentage.csv', encoding='UTF-8', low_memory=False)

# Получение уникальных значений для каждого столбца
unique_values = {col: train_data_cleaned[col].unique() for col in train_data_cleaned.columns}
unique_values_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in unique_values.items()]))
unique_values_df.dropna(inplace=True)

# Количество строк, которые нужно синтезировать для каждой возрастной категории
num_rows_to_generate_per_group = 25000  # Половина для группы 0, половина для группы 3
total_rows_to_generate = num_rows_to_generate_per_group * 2  # Всего 50,000 строк

# Функция для генерации данных
def generate_data(probabilities_df, age_class):
    generated_rows = []
    for _ in range(num_rows_to_generate_per_group):
        selected_row = probabilities_df.sample(weights=probabilities_df['percentage'], n=1).iloc[0]
        event_timestamp = "2024-06-01 03:40:58+00:00"  # Здесь можно использовать случайную дату
        region = random.choice(unique_values_df['region'].dropna())
        ua_device_type = random.choice(unique_values_df['ua_device_type'].dropna())
        ua_client_type = random.choice(unique_values_df['ua_client_type'].dropna())
        ua_os = random.choice(unique_values_df['ua_os'].dropna())
        ua_client_name = random.choice(unique_values_df['ua_client_name'].dropna())
        total_watchtime = random.randint(1000, 3000000)
        viewer_uid = random.randint(1000000, 9999999)
        title = f"Video Title {random.randint(1, 100)}"
        author_id = random.randint(1000000, 9999999)
        age = random.randint(18, 60)
        generated_row = [
            event_timestamp, region, ua_device_type, ua_client_type, ua_os, ua_client_name,
            total_watchtime, "video_" + str(random.randint(100000, 999999)), viewer_uid,
            age, selected_row['sex'], age_class, title, selected_row['category'],
            selected_row['duration'], author_id, selected_row['hour'],
            selected_row['day_of_week'], selected_row['month'], selected_row['season'],
            "long_watch", random.randint(1000, 500000), random.randint(0, 1000)
        ]
        generated_rows.append(generated_row)
    return generated_rows

# Генерация данных для возрастных категорий
generated_data = []
generated_data += generate_data(probabilities_age_0_df, age_class=0)
generated_data += generate_data(probabilities_age_3_df, age_class=3)

# Преобразование в DataFrame
generated_df = pd.DataFrame(generated_data, columns=[
    'event_timestamp', 'region', 'ua_device_type', 'ua_client_type',
    'ua_os', 'ua_client_name', 'total_watchtime', 'rutube_video_id',
    'viewer_uid', 'age', 'sex', 'age_class', 'title', 'category',
    'duration', 'author_id', 'hour', 'day_of_week', 'month',
    'season', 'watch_time_category', 'total_watchtime_user_activity', 'view_count'
])

# Объединяем с основным набором данных
train_data_cleaned = pd.concat([train_data_cleaned, generated_df], ignore_index=True)

# Подсчёт количества мужчин и женщин в каждом городе
sex_counts_per_city = train_data_cleaned.groupby(['region', 'sex']).size().unstack(fill_value=0)

# Подсчёт вероятности быть женщиной в каждом городе
sex_counts_per_city['prob_woman'] = sex_counts_per_city['female'] / (sex_counts_per_city['male'] + sex_counts_per_city['female'])

# Присоединение вероятности к основному датасету
train_data_cleaned = train_data_cleaned.merge(sex_counts_per_city[['prob_woman']], on='region', how='left')

# Группировка данных по категориям и полу
category_sex_counts = train_data_cleaned.groupby(['category', 'sex']).size().unstack(fill_value=0)
category_sex_counts['prob_woman'] = category_sex_counts['female'] / (category_sex_counts['female'] + category_sex_counts['male'])
train_data_cleaned = train_data_cleaned.merge(category_sex_counts[['prob_woman']], on='category', how='left')
train_data_cleaned.rename(columns={'prob_woman': 'prob_woman_categories'}, inplace=True)

# Подсчет общего количества пользователей по городам и возрастным категориям
age_counts_cities = train_data_cleaned.groupby(['region', 'age_class']).size().unstack(fill_value=0)
probabilities_cities = age_counts_cities.div(age_counts_cities.sum(axis=1), axis=0)
for age_class in probabilities_cities.columns:
    train_data_cleaned[f'prob_{age_class}_cities'] = train_data_cleaned['region'].map(probabilities_cities[age_class])

# Подсчет общего количества пользователей по категориям видео и возрастным категориям
age_counts_categories = train_data_cleaned.groupby(['category', 'age_class']).size().unstack(fill_value=0)
probabilities_categories = age_counts_categories.div(age_counts_categories.sum(axis=1), axis=0)
for age_class in probabilities_categories.columns:
    train_data_cleaned[f'prob_{age_class}_categories'] = train_data_cleaned['category'].map(probabilities_categories[age_class])

# Группировка по ua_device_type и полу
sex_counts_device = train_data_cleaned.groupby(['ua_device_type', 'sex']).size().unstack(fill_value=0)
prob_female_device = sex_counts_device['female'] / (sex_counts_device['female'] + sex_counts_device['male'])

# Группировка по ua_client_type и полу
sex_counts_client = train_data_cleaned.groupby(['ua_client_type', 'sex']).size().unstack(fill_value=0)
prob_female_client = sex_counts_client['female'] / (sex_counts_client['female'] + sex_counts_client['male'])

# Группировка по ua_os и полу
sex_counts_os = train_data_cleaned.groupby(['ua_os', 'sex']).size().unstack(fill_value=0)
prob_female_os = sex_counts_os['female'] / (sex_counts_os['female'] + sex_counts_os['male'])

# Группировка по ua_client_name и полу
sex_counts_client_name = train_data_cleaned.groupby(['ua_client_name', 'sex']).size().unstack(fill_value=0)
prob_female_client_name = sex_counts_client_name['female'] / (sex_counts_client_name['female'] + sex_counts_client_name['male'])

# Добавляем в датафрейм вероятности
train_data_cleaned['prob_female_device'] = train_data_cleaned['ua_device_type'].map(prob_female_device)
train_data_cleaned['prob_female_client'] = train_data_cleaned['ua_client_type'].map(prob_female_client)
train_data_cleaned['prob_female_os'] = train_data_cleaned['ua_os'].map(prob_female_os)
train_data_cleaned['prob_female_client_name'] = train_data_cleaned['ua_client_name'].map(prob_female_client_name)

# Список возрастных категорий
age_classes = train_data_cleaned['age_class'].unique()

# Группировка по ua_device_type и возрастной категории
age_counts_device = train_data_cleaned.groupby(['ua_device_type', 'age_class']).size().unstack(fill_value=0)
prob_age_device = age_counts_device.div(age_counts_device.sum(axis=1), axis=0)

# Группировка по ua_client_type и возрастной категории
age_counts_client = train_data_cleaned.groupby(['ua_client_type', 'age_class']).size().unstack(fill_value=0)
prob_age_client = age_counts_client.div(age_counts_client.sum(axis=1), axis=0)

# Группировка по ua_os и возрастной категории
age_counts_os = train_data_cleaned.groupby(['ua_os', 'age_class']).size().unstack(fill_value=0)
prob_age_os = age_counts_os.div(age_counts_os.sum(axis=1), axis=0)

# Группировка по ua_client_name и возрастной категории
age_counts_client_name = train_data_cleaned.groupby(['ua_client_name', 'age_class']).size().unstack(fill_value=0)
prob_age_client_name = age_counts_client_name.div(age_counts_client_name.sum(axis=1), axis=0)

# Добавляем в датафрейм вероятности
for age_class in age_classes:
    train_data_cleaned[f'prob_{age_class}_device'] = train_data_cleaned['ua_device_type'].map(prob_age_device[age_class])
    train_data_cleaned[f'prob_{age_class}_client'] = train_data_cleaned['ua_client_type'].map(prob_age_client[age_class])
    train_data_cleaned[f'prob_{age_class}_os'] = train_data_cleaned['ua_os'].map(prob_age_os[age_class])
    train_data_cleaned[f'prob_{age_class}_client_name'] = train_data_cleaned['ua_client_name'].map(prob_age_client_name[age_class])

# Удаляем строки с пустыми значениями во всех столбцах
train_data_cleaned.dropna(inplace=True)
print(train_data_cleaned.columns)

batch_size=10000

# Инициализация векторизатора
vectorizer = TfidfVectorizer(min_df=500)

# Обработка первого батча отдельно
first_batch = train_data_cleaned['title'].iloc[0:batch_size]
tfidf_matrix = vectorizer.fit_transform(first_batch)

# Обработка остальных батчей
for i in range(batch_size, len(train_data_cleaned), batch_size):
    batch = train_data_cleaned['title'].iloc[i:i + batch_size]
    tfidf_batch = vectorizer.transform(batch)  # Используйте transform вместо fit_transform
    tfidf_matrix = vstack([tfidf_matrix, tfidf_batch])  # Объединение разреженных матриц

# Получение названий фичей (слов)
feature_names = vectorizer.get_feature_names_out()

# Создание DataFrame из разреженной матрицы TF-IDF
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Объединение с основным датасетом
train_data_cleaned = pd.concat([train_data_cleaned.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# Сохранение векторизатора
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Удаление столбца 'event_timestamp'
train_data_cleaned.drop(columns=['event_timestamp'], inplace=True, errors='ignore')
print("Столбец 'event_timestamp' удалён.")

# Сохранение в CSV
train_data_cleaned.to_csv('train_data_cleaned.csv', index=False)