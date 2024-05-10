import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imbalance_pipeline

# Load data
file_path = 'C:/Users/mmira/Desktop/ARIZONA/Capstone_Project/processed_data.csv'
data = pd.read_csv(file_path)
X = data.drop('stunting', axis=1)
y = data['stunting']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and evaluate models
def evaluate_models(models, X_train, y_train, X_test, y_test):
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Evaluating {name}...")
        print(classification_report(y_test, y_pred))

# Create and apply SMOTE
smote = SMOTE(random_state=42)
models = {
    'Logistic Regression': make_imbalance_pipeline(StandardScaler(), smote, LogisticRegression(random_state=42)),
    'Random Forest': make_imbalance_pipeline(StandardScaler(), smote, RandomForestClassifier(random_state=42)),
    'SVC': make_imbalance_pipeline(StandardScaler(), smote, SVC(random_state=42)),
    'KNN': make_imbalance_pipeline(StandardScaler(), smote, KNeighborsClassifier())
}

# Cross-validation and evaluation
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
    print(f"{name} - CV F1 Score: {cv_scores.mean()}")

evaluate_models(models, X_train, y_train, X_test, y_test)

# Extracting feature importance from Logistic Regression and Random Forest
if 'Logistic Regression' in models:
    lr_model = models['Logistic Regression'].named_steps['logisticregression']
    lr_coefs = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr_model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)
    print("Logistic Regression Coefficients:")
    print(lr_coefs)

if 'Random Forest' in models:
    rf_model = models['Random Forest'].named_steps['randomforestclassifier']
    rf_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("Random Forest Feature Importances:")
    print(rf_importances)
