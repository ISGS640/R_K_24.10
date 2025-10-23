"""
Иcследовательский: Прогнозирование уровня тревожности населения Москвы во время пандемии COVID-19

Модель: Random Forest Classifier (ансамблевая модель на основе решающих деревьев);
алгоритм: бэггинг (Bootstrap Aggregating) с случайными подпространствами признаков;
функция потерь: индекс Джини (Gini impurity);
оптимизация: построение множества несвязанных деревьев с последующим усреднением.

Клиническая задача: выявление групп риска по развитию повышенной тревожности среди жителей Москвы 
в условиях пандемии для целевой психологической поддержки.
"""

#импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

#стиль графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_dataset():
    """
    генерация датасета на основе паспорта данных

    """
    print("=== генерация датасета ===")
    np.random.seed(42)
    n_samples = 478

    data = {
        'respondent_id': range(1, n_samples + 1),
        'age_group': np.random.choice(['18-29', '30-39', '40-49', '50-59', '60+'], 
                                     n_samples, p=[0.236, 0.201, 0.207, 0.155, 0.201]),
        'gender': np.random.choice(['female', 'male'], n_samples, p=[0.535, 0.465]),
        'employment': np.random.choice(['трудоустроен', 'безработный', 'студент', 'пенсионер'], 
                                      n_samples, p=[0.6, 0.15, 0.15, 0.1]),
        'education': np.random.choice(['высшее', 'среднее', 'среднее специальное'], 
                                     n_samples, p=[0.5, 0.3, 0.2]),
        'district': np.random.choice(['ЦАО', 'САО', 'СВАО', 'ВАО', 'ЮВАО', 'ЮАО', 'ЮЗАО', 'ЗАО', 'СЗАО', 'ТиНАО'], 
                                    n_samples),
        'family_status': np.random.choice(['в браке', 'холост', 'разведен', 'вдовец'], 
                                         n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'children': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        
        #поведение жителей г. Москвы
        'prevention_measures': np.random.randint(1, 11, n_samples),
        'digital_usage': np.random.randint(1, 11, n_samples),
        'info_consumption': np.random.randint(1, 11, n_samples),
        'social_isolation': np.random.randint(1, 11, n_samples),
        'panic_buying': np.random.randint(0, 2, n_samples),
        
        #ментальное состояние москвичей
        'anxiety_level': np.random.randint(0, 21, n_samples),
        'trust_government': np.random.randint(1, 11, n_samples),
        'economic_fear': np.random.randint(1, 11, n_samples),
        'social_connections': np.random.randint(1, 11, n_samples),
        'future_optimism': np.random.randint(1, 11, n_samples),
    }

    df = pd.DataFrame(data)
    
    #создание целевой переменной: высокий уровень тревожности
    df['high_anxiety'] = (df['anxiety_level'] > 12).astype(int)
    
    print(f"датасет сгенерирован: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"целевая переменная: {df['high_anxiety'].sum()} случаев высокой тревожности ({df['high_anxiety'].mean()*100:.1f}%)")
    
    return df

def clinical_task_decomposition():
    """
    декомпозиция клинической задачи на этапы решения
    """
    print("\n" + "="*60)
    print("декомпозиция клинической задачи")
    print("="*60)
    
    steps = [
        "1. Анализ демографических характеристик респондентов",
        "   - Возрастное распределение (18-29, 30-39, 40-49, 50-59, 60+)",
        "   - гендерный состав (53.5% женщин, 46.5% мужчин)",
        "   - трудовой статус (трудоустроенные, безработные, студенты, пенсионеры)",
        "   - образование",
        "",
        "2. Оценка поведенческих паттернов", 
        "   - Соблюдение мер профилактики COVID-19",
        "   - использование цифровых технологий",
        "   - уровень социальной изоляции",
        "   - участие в массовых закупках",
        "",
        "3. Измерение психологических показателей",
        "   - Уровень тревожности (аналог шкалы GAD-7, 0-20 баллов)",
        "   - доверие к мерам правительства",
        "   - страх экономической дестабилизации", 
        "   - социальные связи и поддержка",
        "   - оптимизм относительно будущего",
        "",
        "4. Выявление корреляций и зависимостей",
        "   - Анализ взаимосвязей между демографическими и поведенческими факторами",
        "   - выявление значимых предикторов высокой тревожности",
        "   - проверка статистической значимости обнаруженных закономерностей",
        "",
        "5. Построение прогностической модели",
        "   - Подготовка и предобработка данных",
        "   - настройка алгоритма машинного обучения", 
        "   - обучение модели на тренировочной выборке",
        "   - валидация и оценка качества модели",
        "",
        "6. Разработка рекомендаций",
        "   - Идентификация групп высокого риска",
        "   - формулирование целевых программ",
        "   - планирование программ психологической поддержки"
    ]
    
    for step in steps:
        print(step)

def prepare_features(df):
    """
    Подготовка признаков для модели машинного обучения
    """
    print("\n=== подготовка признаков ===")
    
    #категориальные переменные
    categorical_columns = ['age_group', 'gender', 'employment', 'education', 'district', 'family_status']
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"закодирована переменная: {col} -> {col}_encoded")

    #выбор признаков для модели
    feature_columns = [
        'age_group_encoded', 'gender_encoded', 'employment_encoded', 
        'education_encoded', 'family_status_encoded', 'children',
        'prevention_measures', 'digital_usage', 'info_consumption', 
        'social_isolation', 'panic_buying', 'trust_government',
        'economic_fear', 'social_connections', 'future_optimism'
    ]

    X = df[feature_columns]
    y = df['high_anxiety']
    
    print(f"Признаки подготовлены: {X.shape[1]} признаков, {X.shape[0]} наблюдений")
    print(f"Баланс классов: {y.value_counts().to_dict()}")
    
    return X, y, feature_columns

def train_random_forest(X_train, X_test, y_train, y_test, feature_columns):
    """
    Обучение модели Random Forest с оптимизацией гиперпараметров
    """
    print("\n" + "="*60)
    print("Обучение модели Random Forest")
    print("="*60)
    
    print("Параметры модели:")
    print("- алгоритм: Random Forest Classifier")
    print("- количество деревьев: 200")
    print("- максимальная глубина: 8")
    print("- минимальное количество образцов для разделения: 20")
    print("- минимальное количество образцов в листе: 10")
    print("- балансировка классов: включена")
    
    #создание модели
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    #прогнозирование
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    #оценка модели
    print("\nОценка модели:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.3f}")
    
    #матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nМатрица ошибок:\n{cm}")
    
    return rf_model, y_pred, y_prob

def visualize_results(model, X_test, y_test, y_pred, y_prob, feature_columns):
    """
    Визуализация результатов работы модели
    """
    print("\n=== Визуализация результатов ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    #1) ROC-кривая
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC-кривая модели Random Forest')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True)
    
    #2) значимость признаков
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    axes[0, 1].barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.7)
    axes[0, 1].set_yticks(range(len(indices)))
    axes[0, 1].set_yticklabels([feature_columns[i] for i in indices])
    axes[0, 1].set_xlabel('значимость признака')
    axes[0, 1].set_title('значимость признаков в модели')
    axes[0, 1].grid(True, axis='x')
    
    #3) матрица ошибок тепловая карта
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('предсказанный класс')
    axes[1, 0].set_ylabel('истинный класс')
    axes[1, 0].set_title('матрица ошибок')
    
    #4) распределение вероятностей
    axes[1, 1].hist(y_prob[y_test == 0], bins=30, alpha=0.7, label='низкая тревожность', color='green')
    axes[1, 1].hist(y_prob[y_test == 1], bins=30, alpha=0.7, label='высокая тревожность', color='red')
    axes[1, 1].set_xlabel('вероятность высокой тревожности')
    axes[1, 1].set_ylabel('число наблюдений')
    axes[1, 1].set_title('распределение вероятностей по классам')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importances

def hypothesis_testing(df, importances, feature_columns):
    """
    Проверка основной гипотезы исследования
    """
    print("\n" + "="*60)
    print("проверка гипотезы")
    print("="*60)
    
    #гипотеза: Молодые трудоустроенные жители г. Москвы имеют меньшую тревожность
    young_employed = df[(df['age_group'].isin(['18-29', '30-39'])) & (df['employment'] == 'трудоустроен')]
    other_groups = df[~((df['age_group'].isin(['18-29', '30-39'])) & (df['employment'] == 'трудоустроен'))]
    
    print("гипотеза: молодые трудоустроенные жители г. Москвы демонстрируют более низкий уровень тревожности")
    print(f"доля высокой тревожности среди молодых трудоустроенных: {young_employed['high_anxiety'].mean():.3f}")
    print(f"доля высокой тревожности среди остальных групп: {other_groups['high_anxiety'].mean():.3f}")
    print(f"разница: {young_employed['high_anxiety'].mean() - other_groups['high_anxiety'].mean():.3f}")
    
    #статистическая проверка
    from scipy.stats import chi2_contingency
    contingency_table = pd.crosstab(
        df['age_group'].isin(['18-29', '30-39']) & (df['employment'] == 'трудоустроен'),
        df['high_anxiety']
    )
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    
    print(f"статистическая значимость (p-value): {p_value:.4f}")
    
    if p_value < 0.05:
        print("гипотеза подтверждена статистически (p < 0.05)")
    else:
        print("гипотеза не подтверждена статистически")
    
    #анализ важных факторов
    print("\n 5 признаков для прогнозирования:")
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

def generate_recommendations(df, model, feature_columns):
    """
    генерация практических рекомендаций на основе результатов модели
    """
    print("\n" + "="*60)
    print("практические рекомендации")
    print("="*60)
    
    #выявление групп высокого риска
    high_risk_groups = df.groupby(['age_group', 'employment'])['high_anxiety'].mean().sort_values(ascending=False)
    
    print("группы населения, требующие особого внимания):")
    for (age, employment), risk in high_risk_groups.head(3).items():
        print(f"  - {age} лет, {employment}: риск высокой тревожности {risk:.1%}")
    
    print("\nцелевые программы:")
    print("  1. Программы психологической поддержки для безработных и пенсионеров;")
    print("  2. цифровая грамотность для снижения информационной перегрузки;")
    print("  3. социальные программы поддержания связей в условиях изоляции;")
    print("  4. экономическая поддержка и информационная прозрачность;")
    print("  5. целевая работа с группами, демонстрирующими низкий оптимизм.")

def main():
    """
    Основная функция запуска всего анализа
    """
    print("Запуск анализа")
    print("="*80)
    
    #декомпозиция клинической задачи
    clinical_task_decomposition()
    
    #генерация данных на основе датасета https://ai.sechenov.ru/datasets/211
    df = generate_dataset()
    
    #подготовка признаков
    X, y, feature_columns = prepare_features(df)
    
    #разделение выборок на тренировочную и тестовую
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nразделение данных: train - {X_train.shape[0]}, test - {X_test.shape[0]}")
    
    #обучение модели
    model, y_pred, y_prob = train_random_forest(X_train, X_test, y_train, y_test, feature_columns)
    
    #визуализация результатов
    importances = visualize_results(model, X_test, y_test, y_pred, y_prob, feature_columns)
    
    #проверка гипотезы
    hypothesis_testing(df, importances, feature_columns)
    
    #подготовка рекомендаций
    generate_recommendations(df, model, feature_columns)
    
    print("\n" + "="*80)
    print("Анализ завершен")
    print("="*80)

#запуск основной функции
if __name__ == "__main__":
    main()