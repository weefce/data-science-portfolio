# Прогнозирование оттока клиентов телеком-компании 

## Описание проекта
Прогнозная модель для выявления клиентов с высоким риском оттока. Цель - проактивное удержание через персонализированные меры

## Бизнес-задача
Снижение оттока клиентов за счет раннего выявления рисковых клиентов и точечных удержательных кампаний

## Ключевые результаты 
- **Recall: 87%** - находим 87% уходящих клиентов 
- **Precision: 50%** - каждый второй предсказанный реально уйдет  
- **AUC-ROC: 0.88** - модель отлично разделяет классы
- **Оптимальный порог: 0.475** - баланс между recall и precision

## Топ-факторы оттока: 
1. **Временный клиент** (короткий tenure + помесячный контракт)
2. **Тип контракта** (Month-to-month vs Two year)
3. **Интернет-провайдер Fiber optic без дополнительных услуг**
4. **Способ оплаты Electronic check**
5. **Отсутствие онлайн сервисов**

## Технологии
- **Библиотеки:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
- **Методы:** OneHotEncoding, StandardScaler, RandomizedSearchCV, ROC-AUC анализ
- **Модели:** LogisticRegression, SVM, RandomForest, XGBoost

## Источник данных 
Проект использует датасет [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data) с Kaggle
Размер: 7043 клиента, 21 признак

## Структура проекта 
01_customer_churn/
  data/
    WA_Fn-UseC_-Telco-Customer-Churn.csv
  notebooks/
    churn_analysis.ipynb
  models/
    xgb_model.pkl
  results/
    predictions.csv
    feature_importance.csv
    feature_importance_plot.png
  README.md
  requirements.txt

## Как запустить 
1. Клонировать репозиторий

```git clone <ссылка-на-репозиторий>
cd customer_churn```

2. Установить зависимости
``` pip install -r requirements.txt```

3. Запустить Jupyter Notebook
```jupyter notebook notebooks/churn_analysis.ipynb```

## Основные этапы работы 
1. EDA - анализ данных, выявление паттернов оттока
2. Feature Engineering - создание признаков: temporary_customer, fiber_without_services
3. Моделирование - сравнение 5 моделей, подбор гиперпараметров
4. Выбор модели - XGBoost с кастомным порогом 0.475
5. Интерпретация - анализ важности признаков
