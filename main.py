import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['amount_bin'] = pd.qcut(df['amount'], q=5, labels=False)
    df['time_bin'] = pd.cut(df['time'], bins=6, labels=False)
    df.drop(['amount', 'time'], axis=1, inplace=True)
    return df

def split_data(df: pd.DataFrame, target: str = 'class'):
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def train_model(X_train, y_train):
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_estimators=100
    )
    model.fit(X_train, y_train, verbose=False)
    return model

def evaluate_model(model, X_valid, y_valid):
    y_pred = model.predict(X_valid)
    y_prob = model.predict_proba(X_valid)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_valid, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_valid, y_prob):.4f}")

def plot_shap(model, X_valid):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_valid)
    shap.summary_plot(shap_values, X_valid, plot_type="bar")

def main():
    df = load_data("creditcard.csv")

    sns.countplot(x='class', data=df)
    plt.title("Fraud vs Non-Fraud")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    df = preprocess_data(df)
    X_train, X_valid, y_train, y_valid = split_data(df)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_valid, y_valid)
    plot_shap(model, X_valid)

if __name__ == "__main__":
    main()
