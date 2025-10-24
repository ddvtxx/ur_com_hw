from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def part3_classification():
    """Part 3: Classification Models"""
    print("\nPART 3: CLASSIFICATION MODELS")

    # Load and prepare data
    df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.set_index('date_time')

    # Create time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Create categorical target (BINNING RULE)
    def create_traffic_class(volume):
        if volume < 1500:
            return 'low'
        elif volume <= 4000:
            return 'medium'
        else:
            return 'high'

    df['traffic_class'] = df['traffic_volume'].apply(create_traffic_class)

    print("=== BINNING RULE ===")
    print("Low: < 1,500 vehicles")
    print("Medium: 1,500 - 4,000 vehicles")
    print("High: > 4,000 vehicles")
    print(f"\nClass distribution:\n{df['traffic_class'].value_counts()}")

    # Define features and target
    X_clf = df.drop(['traffic_volume', 'traffic_class'], axis=1)
    y_clf = df['traffic_class']

    # Encode string labels to integers for XGBoost compatibility
    label_encoder = LabelEncoder()
    y_clf_encoded = label_encoder.fit_transform(y_clf)

    # Map the encoding for interpretation
    class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    print(f"\nLabel encoding: {class_mapping}")

    # Chronological split
    split_index = int(0.8 * len(X_clf))
    X_train_clf, X_test_clf = X_clf.iloc[:split_index], X_clf.iloc[split_index:]
    y_train_clf, y_test_clf = y_clf_encoded[:split_index], y_clf_encoded[split_index:]

    # Preprocessing
    numeric_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
    categorical_features = ['holiday', 'weather_main', 'is_weekend']
    cyclical_features = ['hour_sin', 'hour_cos']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('pass', 'passthrough', cyclical_features + ['dayofweek', 'month', 'year'])
        ])

    # Define classification models - FIXED: All models now use encoded labels
    clf_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, solver='saga', penalty='l2', C=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
    }

    clf_results = {}
    tscv = TimeSeriesSplit(n_splits=5)  # k=5 cross-validation

    for name, model in clf_models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Cross-validation
        try:
            cv_scores = cross_val_score(pipeline, X_train_clf, y_train_clf, cv=tscv,
                                        scoring='accuracy')

            # Train and predict
            pipeline.fit(X_train_clf, y_train_clf)
            y_pred_clf = pipeline.predict(X_test_clf)

            accuracy = accuracy_score(y_test_clf, y_pred_clf)

            clf_results[name] = {
                'CV_Accuracy_mean': cv_scores.mean(),
                'CV_Accuracy_std': cv_scores.std(),
                'Test_Accuracy': accuracy
            }

            print(f"{name} Classification Report:")
            print(classification_report(y_test_clf, y_pred_clf,
                                        target_names=label_encoder.classes_))

        except Exception as e:
            print(f"Error training {name}: {e}")
            clf_results[name] = {
                'CV_Accuracy_mean': np.nan,
                'CV_Accuracy_std': np.nan,
                'Test_Accuracy': np.nan
            }

    # Display results
    clf_results_df = pd.DataFrame(clf_results).T
    print("\n=== CLASSIFICATION RESULTS ===")
    print(clf_results_df.round(4))

    # Confusion matrix for best model
    best_clf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, random_state=42))
    ])
    best_clf_pipeline.fit(X_train_clf, y_train_clf)
    y_pred_best = best_clf_pipeline.predict(X_test_clf)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_clf, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - XGBoost Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    return clf_results_df


# Run Part 3
classification_results = part3_classification()