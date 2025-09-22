import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)

# =====================================================
# 1. DATA UNDERSTANDING
# =====================================================
df = pd.read_csv("C:/Users/syadi/Documents/Magang IDX Partners/loan_data_2007_2014.csv")

print("=== Info Data ===")
print(df.info())
print("\nJumlah baris dan kolom:", df.shape)

print("\n=== Statistik Deskriptif ===")
print(df.describe(include="all").transpose())

print("\n=== Distribusi Loan Status ===")
print(df['loan_status'].value_counts())

# Fokus hanya Fully Paid vs Charged Off/Default
df = df[df['loan_status'].isin(['Fully Paid','Charged Off','Default'])]
df['label'] = df['loan_status'].apply(lambda x: 0 if x=="Fully Paid" else 1)

# =====================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================
sns.countplot(x="label", data=df)
plt.title("Distribusi Good vs Bad Loan")
plt.show()

sns.histplot(df["int_rate"], kde=True)
plt.title("Distribusi Interest Rate")
plt.show()

sns.boxplot(x="label", y="annual_inc", data=df)
plt.yscale("log")
plt.title("Annual Income vs Loan Status")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(
    df[["loan_amnt","funded_amnt","int_rate","installment",
        "annual_inc","dti","delinq_2yrs","open_acc","pub_rec","label"]].corr(),
    annot=True, cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# =====================================================
# 3. DATA PREPARATION
# =====================================================
features = [
    "loan_amnt","funded_amnt","term","int_rate","installment",
    "grade","sub_grade","emp_length","home_ownership","annual_inc",
    "verification_status","purpose","dti","delinq_2yrs",
    "inq_last_6mths","open_acc","pub_rec"
]
X = df[features]
y = df['label']

# Handle missing values
X = X.fillna({
    col: X[col].median() if X[col].dtype != 'object' else X[col].mode()[0]
    for col in X.columns
})

# Encoding kategori
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Normalisasi numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# 4. DATA MODELLING
# =====================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

# =====================================================
# 5. EVALUATION
# =====================================================
results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

plt.figure(figsize=(8,6))  # ROC Curve Comparison

for name, model in models.items():
    print(f"\n=== {name} ===")
    
    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    # Metrics (ubah ke persen)
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_prob) * 100
    cv_acc = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy").mean() * 100

    results.append({
        "Model": name,
        "Accuracy (%)": acc,
        "Precision (%)": prec,
        "Recall (%)": rec,
        "F1-score (%)": f1,
        "ROC-AUC (%)": auc,
        "CV Accuracy (%)": cv_acc
    })

    # Classification Report
    print(classification_report(y_test, y_pred, target_names=["Good Loan", "Bad Loan"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Pred Good", "Pred Bad"], 
                yticklabels=["True Good", "True Bad"])
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f}%)")

# Ringkasan hasil
results_df = pd.DataFrame(results)
print("\n=== Ringkasan Evaluasi Model ===")
print(results_df)

# ROC Curve Comparison
plt.figure(figsize=(8,6))  # buat canvas sekali di awal

for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2%})")

# garis diagonal
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

