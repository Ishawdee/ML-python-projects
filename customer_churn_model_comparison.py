# Customer churn prediction using different classical ML algorithms.
# Goal: predict whether a customer will leave the company.
# This project compares Logistic Regression, KNN, Decision Tree, and Random Forest.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score
)


# 1. Reusable evaluation function
# It trains the given model, makes predictions, prints useful metrics,
# and optionally shows a confusion matrix.
# Since this dataset is imbalanced, we care about more than accuracy:
# precision, recall, F1, ROC-AUC, & PR-AUC are also important.
# ============================================================

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name,
    show_matrix=False
):
    """
    Fits a model, evaluates it on train/test data, prints metrics,
    and optionally shows a confusion matrix.
    """

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred)
    test_recall = recall_score(y_test, test_pred)

    print("\n" + "=" * 60)
    print(f"{model_name} results")
    print("=" * 60)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Churn precision: {test_precision:.4f}")
    print(f"Churn recall:    {test_recall:.4f}")
    print(f"Churn F1 score:  {test_f1:.4f}")

    roc_auc = None
    pr_auc = None

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC:  {pr_auc:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        test_pred,
        target_names=["No churn", "Churn"]
    ))

    if show_matrix:
        cm = confusion_matrix(y_test, test_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No churn", "Churn"]
        )

        disp.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        plt.show()

    results = {
        "model": model_name,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "churn_precision": test_precision,
        "churn_recall": test_recall,
        "churn_f1": test_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

    return results


# 2. Load dataset
# ============================================================

base_path = Path(__file__).parent
data_path = base_path / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(data_path)

print("Original shape:", df.shape)


# 3. Clean data
# ============================================================

# TotalCharges is loaded as a string because some rows contain blank spaces.
# Convert invalid values to NaN, then drop those rows.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# customerID is just an identifier, so not a useful predictive feature.
df = df.drop(columns=["customerID"])

# Convert target labels:
# No churn -> 0
# Churn    -> 1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

print("Shape after cleaning:", df.shape)
print("\nChurn distribution:")
print(df["Churn"].value_counts())


# 4. Split features and target
# ============================================================

X = df.drop(columns=["Churn"])
y = df["Churn"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns

print("\nNumeric features:")
print(list(numeric_features))

print("\nCategorical features:")
print(list(categorical_features))


# 5. Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)
# Numeric features need scaling, especially for Logistic Regression and KNN.
# Categorical features need one-hot encoding, ML models can't directly use text labels.

# 6. Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# 7. Model experiments
# ============================================================

all_results = []

SHOW_MATRICES = False


# 7.1 Logistic Regression baseline
# ------------------------------------------------------------

log_reg_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

result = evaluate_model(
    log_reg_model,
    X_train,
    X_test,
    y_train,
    y_test,
    "Logistic Regression",
    show_matrix=SHOW_MATRICES
)

all_results.append(result)

# Train accuracy: 0.8046
# Test accuracy:  0.8038
# Churn F1:       0.6080
# ROC-AUC:        0.8359

# 7.2 Logistic Regression with class_weight="balanced"
# ------------------------------------------------------------

log_reg_balanced_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]
)

result = evaluate_model(
    log_reg_balanced_model,
    X_train,
    X_test,
    y_train,
    y_test,
    "Logistic Regression Balanced",
    show_matrix=SHOW_MATRICES
)

all_results.append(result)

# Train accuracy: 0.7556
# Test accuracy:  0.7257
# Churn precision: 0.49
# Churn recall:    0.80
# Churn F1:        0.6069
# ROC-AUC:         0.8351


# 7.3 KNN with different k values
# ------------------------------------------------------------
# Small k can overfit, large k can underfit.
k_values = [3, 5, 9, 15, 25]

for k in k_values:
    knn_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=k))
        ]
    )

    result = evaluate_model(
        knn_model,
        X_train,
        X_test,
        y_train,
        y_test,
        f"KNN k={k}",
        show_matrix=SHOW_MATRICES
    )

    all_results.append(result)

# Previous KNN best result was around k=25:
# Test accuracy: 0.7825
# Churn F1:      0.5887
# ROC-AUC:       0.8255
# Logistic Regression performed better than KNN for this task.

# 7.4 Decision Tree with different max_depth values
# ------------------------------------------------------------

depth_values = [3, 5, 7, None]

for depth in depth_values:
    decision_tree_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=20,
                random_state=42
            ))
        ]
    )

    result = evaluate_model(
        decision_tree_model,
        X_train,
        X_test,
        y_train,
        y_test,
        f"Decision Tree max_depth={depth}",
        show_matrix=SHOW_MATRICES
    )

    all_results.append(result)


# 7.5 Random Forest
# ------------------------------------------------------------

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

result = evaluate_model(
    random_forest_model,
    X_train,
    X_test,
    y_train,
    y_test,
    "Random Forest",
    show_matrix=SHOW_MATRICES
)

all_results.append(result)


# 7.6 Random Forest with class_weight="balanced"
# ------------------------------------------------------------

random_forest_balanced_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]
)

result = evaluate_model(
    random_forest_balanced_model,
    X_train,
    X_test,
    y_train,
    y_test,
    "Random Forest Balanced",
    show_matrix=SHOW_MATRICES
)

all_results.append(result)


# 8. Summary table
# ============================================================

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 60)
print("Model comparison summary")
print("=" * 60)

print(
    results_df.sort_values(
        by="churn_f1",
        ascending=False
    ).to_string(index=False)
)

# 9. Show confusion matrix for the best model by churn F1
# ============================================================

best_model_name = results_df.sort_values(
    by="churn_f1",
    ascending=False
).iloc[0]["model"]

print("\nBest model by Churn F1:", best_model_name)

# best model ended up being random forest balanced:
# Random Forest Balanced
# Churn precision: 0.5158
# Churn recall:    0.7834
# Churn F1:        0.6221
# ROC-AUC:         0.8357
# PR-AUC:          0.6400