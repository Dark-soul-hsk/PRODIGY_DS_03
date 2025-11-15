import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


FILE_PATH = r"C:\Users\Hemant Sri Kumar\Desktop\Prodigy\Task_03\bank\bank.csv"


try:
    df = pd.read_csv(FILE_PATH, sep=';')
except FileNotFoundError:
    print(f"Error: File not found at the specified path: {FILE_PATH}")
    exit()

df['y'] = df['y'].map({'no': 0, 'yes': 1})


categorical_features = df.select_dtypes(include='object').columns.tolist()

df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)


X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)

dt_classifier.fit(X_train, y_train)


y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n=============================================")
print(f"Model Accuracy: {accuracy:.4f}")
print("=============================================")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Subscription', 'Subscription']))


feature_importance = pd.Series(dt_classifier.feature_importances_, index=X.columns)
top_10_features = feature_importance.nlargest(10)

print("\n--- Top 10 Most Important Features ---")
print(top_10_features.to_markdown(floatfmt=".4f"))


