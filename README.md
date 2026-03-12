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

## Технологический стек

- Язык: Python 3.9+
- Анализ данных: Pandas, NumPy
- Машинное обучение: Scikit-learn, XGBoost, CatBoost, LightGBM
- Временные ряды: Statsmodels, pmdarima, Prophet
- Визуализация: Matplotlib, Seaborn
- Инструменты: Git, Jupyter Notebook

---

## Контакты
- Telegram: [@weefce]
- Email: valeri05.lera@mail.ru