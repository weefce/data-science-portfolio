# Data Science Projects Portfolio

Репозиторий с реализованными проектами по машинному обучению и анализу данных.

---

## Структура репозитория

```text 
data-science-portfolio/
    01_customer_churn/               # Прогнозирование оттока клиентов
        data/
            WA_Fn-UseC_-Telco-Customer-Churn.csv
        notebooks/
            churn_analysis.ipynb
        models/
            logreg_model.pkl
        results/
            predictions.csv
            importance.csv
        README.md

    02_bitcoin_forecasting/          # Прогнозирование цены биткоина
        data/
            Bitcoin_kaggle.csv
        notebooks/
            bitcoin_forecasting.ipynb
        models/
            holt_winters_model.pkl
        results/
            predictions.csv
        README.md

    03_amazon_reviews_sentiment/     # Классификация тональности отзывов Amazon
        notebooks/
            amazon_sentiment_analysis.ipynb
        models/
            amazon_pipeline.pkl
        results/
            predictions.csv
        README.md

    04_question_classification/      # Классификация происхождения вопросов (VK)
        data/
            data.csv                 
            train.csv                
            test.csv                
        notebooks/
            classification_questions.ipynb
        models/
            best_xgboost.pkl
        results/
            test_predictions.csv
        README.md

    05_wildberries_hackathon/          # Прогнозирование отгрузок (Wildberries)
        data/
            train_solo_track.parquet
            test_solo_track.parquet
        notebooks/
            wildberries_forecasting.ipynb
        results/
            submission_ridge_7days.csv
        README.md
    
    certificates/                    # Сертификаты
        vk_question_classification.pdf
        stepik_machine_learning_course.pdf
        stepik_sql_course.pdf

    requirements.txt
    README.md
```

## Проекты 

### 1. [Прогнозирование оттока клиентов телеком-компании](01_customer_churn/)

**Задача:** Построить модель для предсказания ухода клиентов телеком-компании

**Что сделано:**
- EDA, проверка гипотез, выявление ключевых факторов оттока
- Feature engineering: temporary_customer, fiber_without_services и др.
- Сравнение моделей: Logistic Regression, SVM, Random Forest, XGBoost, CatBoost
- Оптимизация порога классификации (recall 85% при precision 52%)
- Анализ важности признаков

**Ключевые результаты:**
- Recall (отток): 85% - находим 85% уходящих клиентов
- Precision (отток): 52% - каждый второй предсказанный реально уходит
- Лучшая модель: Logistic Regression с порогом 0.474
- Топ-факторы: Fiber optic, временный клиент, помесячный контракт

**Технологии:** Python, Pandas, Scikit-learn, XGBoost, CatBoost

---

### 2. [Прогнозирование цены биткоина на 7 дней](02_bitcoin_forecasting/)

**Задача:** Сравнить статистические модели и ML-подходы для прогноза цены биткоина с акцентом на избежание утечки данных

**Что сделано:**
- Полный анализ временного ряда (ADF, ACF/PACF, декомпозиция)
- Статистические модели: Holt, Holt-Winters, ARIMA, auto_arima, Prophet
- ML-подход с лагами >= 7 дней (Linear Regression, Ridge, Lasso, XGBoost)
- Сравнение на горизонтах 7, 14 и 30 дней

**Ключевые результаты:**
- Лучшая модель: Holt-Winters (mul-mul) с MAE = 389 (1,44 % ошибки)
- Статистические модели превосходят наивный прогноз (последняя цена)
- ML-модели показали ошибку в 10 раз больше из-за ограничений на утечку

**Технологии:** Python, Statsmodels, pmdarima, Prophet, Scikit-learn, XGBoost

---

### 3. [Классификация тональности отзывов Amazon](03_amazon_reviews_sentiment/)

**Задача:** Автоматическое определение тональности отзывов (позитив/негатив) по тексту

**Что сделано:**
- EDA: анализ разреженности данных (69% продуктов с 1 отзывом), проверка гипотез
- Feature engineering: отбор только текстовых признаков
- TF-IDF векторизация, ColumnTransformer, Pipeline
- Сравнение моделей: Logistic Regression, Naive Bayes, Random Forest, LinearSVC
- Анализ ошибок (false positive / false negative)

**Ключевые результаты:**
- Accuracy: 88%
- F1 (positive): 0.92, F1 (negative): 0.75
- Recall (negative): 83% - находим 83% негативных отзывов
- Лучшая модель: Logistic Regression с балансировкой классов

**Технологии:** Python, Pandas, Scikit-learn (TfidfVectorizer, CountVectorizer)

---

### 4. [Классификация происхождения вопросов (проект VK)](04_question_classification/)

**Задача:** Разработать модель, отличающую вопросы, подготовленные профессиональными редакторами, от вопросов, предложенных пользователями игры

**Что сделано:**
- Полный цикл ML: EDA, TF-IDF, Pipeline, кросс-валидация
- Сравнение 4 моделей: Logistic Regression, XGBoost, Linear SVC, LightGBM
- Подбор гиперпараметров (RandomizedSearchCV)
- Оптимизация порога классификации для повышения recall редкого класса

**Ключевые результаты:** 
- Recall (редакторы): 50% - найдена половина редакторских вопросов
- Recall (пользователи): 71%
- Accuracy: 69%
- Лучшая модель: XGBoost с подбором гиперпараметров

**Сложность задачи:**
- Данные сильно несбалансированы (90% пользователей / 10% редакторов)
- Лексика классов почти неразличима (топ-10 слов одинаков)
- Задача объективно сложная даже для человека
- Методы TF-IDF + XGBoost дали recall 50% - это хороший результат для таких данных

**Технологии:** Python, Pandas, Scikit-learn, XGBoost, NLTK

---

### 5. [Прогнозирование отгрузок со складов (Wildberries)](05_wildberries_hackathon/)

**Задача:** Прогнозирование объёмов отгрузок со складов на 8 шагов вперёд (шаг 30 минут)

**Что сделано:**
- Глубокий EDA, выявление шума и волатильности данных
- Feature Engineering: лаги, суммы, временные признаки
- Валидация на разных временных окнах (7, 14, 21 день)
- Обучение Ridge регрессии на последних 7 днях

**Ключевые результаты:**
- Лучшая модель: Ridge регрессия на последних 7 днях
- Скор на публичном лидерборде: 0.3365
- Скор на приватном лидерборде: 0.3349

**Технологии:** Python, Pandas, Scikit-learn (Ridge), Matplotlib, Seaborn

---

## Технологический стек

- Язык: Python 3.9+
- Анализ данных: Pandas, NumPy
- Машинное обучение: Scikit-learn, XGBoost, CatBoost, LightGBM
- Временные ряды: Statsmodels, pmdarima, Prophet
- Визуализация: Matplotlib, Seaborn
- Инструменты: Git, Jupyter Notebook

---

## Контакты
- Telegram: @weefce
- Email: valeri05.lera@mail.ru