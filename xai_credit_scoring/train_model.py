import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
import xgboost as xgb
import lightgbm as lgb
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DOWNLOAD & PREPARE DATA
# ============================================================
print("=" * 60)
print("  CREDIT SCORING MODEL TRAINING PIPELINE")
print("=" * 60)

print("\n📥 Step 1: Downloading German Credit dataset from UCI...")
columns = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_status', 'employment', 'installment_commitment', 'personal_status',
    'other_parties', 'residence_since', 'property_magnitude', 'age',
    'other_payment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'own_telephone', 'foreign_worker', 'target'
]
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
df = pd.read_csv(url, sep=' ', header=None, names=columns)

# ============================================================
# 2. PREPROCESSING
# ============================================================
print("🔧 Step 2: Preprocessing data...")

# Target: 1 = Good Credit -> 0, 2 = Bad Credit/Default -> 1
df['target'] = df['target'].apply(lambda x: 0 if x == 1 else 1)

# Store label encoders for each categorical column so app.py can decode/encode
encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save processed data and encoders
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
df.to_csv('data/processed_credit_data.csv', index=False)

with open('models/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Save feature names and their categories for the app
feature_info = {}
for col in df.columns:
    if col == 'target':
        continue
    feature_info[col] = {
        'min': int(df[col].min()),
        'max': int(df[col].max()),
        'mean': round(float(df[col].mean()), 2),
        'dtype': 'categorical' if col in encoders else 'numerical'
    }
    if col in encoders:
        feature_info[col]['classes'] = list(encoders[col].classes_)

with open('models/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print(f"   ✅ Data shape: {df.shape}")
print(f"   ✅ Target distribution: {dict(df['target'].value_counts())}")

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

feature_names = list(X.columns)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print(f"   Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ============================================================
# 4. MULTI-MODEL COMPARISON WITH HYPERPARAMETER TUNING
# ============================================================
print("\n🚀 Step 3: Training & comparing 4 models with hyperparameter tuning...\n")

models_config = {
    'XGBoost': {
        'model': xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0
        ),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
        }
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [15, 31, 63],
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
        }
    }
}

results = {}
best_overall_score = 0
best_overall_model = None
best_overall_name = ""

for name, config in models_config.items():
    print(f"  🔄 Training {name}...")

    grid = GridSearchCV(
        config['model'],
        config['params'],
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Cross-validation score
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='roc_auc')

    results[name] = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc, 4),
        'cv_mean_auc': round(cv_scores.mean(), 4),
        'cv_std_auc': round(cv_scores.std(), 4),
        'best_params': grid.best_params_
    }

    print(f"     ✅ Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC-ROC: {auc:.4f} | CV-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Use AUC-ROC as primary metric for selecting best model
    if auc > best_overall_score:
        best_overall_score = auc
        best_overall_model = best_model
        best_overall_name = name

# ============================================================
# 5. SAVE BEST MODEL & REPORT
# ============================================================
print(f"\n{'=' * 60}")
print(f"  🏆 BEST MODEL: {best_overall_name}")
print(f"     AUC-ROC: {best_overall_score:.4f}")
print(f"     Params: {results[best_overall_name]['best_params']}")
print(f"{'=' * 60}")

# Save the best model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_overall_model, f)

# Also save in models/ directory
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_overall_model, f)

# Save comparison report
report = {
    'best_model_name': best_overall_name,
    'best_model_auc': best_overall_score,
    'all_results': results
}
with open('models/model_comparison_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n📊 Model Comparison Report:")
print("-" * 60)
print(f"{'Model':<22} {'Accuracy':>10} {'F1':>8} {'AUC-ROC':>10} {'CV-AUC':>10}")
print("-" * 60)
for name, metrics in results.items():
    marker = " 🏆" if name == best_overall_name else ""
    print(f"{name:<22} {metrics['accuracy']:>10.4f} {metrics['f1_score']:>8.4f} {metrics['auc_roc']:>10.4f} {metrics['cv_mean_auc']:>10.4f}{marker}")
print("-" * 60)

print(f"\n✅ All artifacts saved:")
print(f"   • model.pkl (best model)")
print(f"   • models/best_model.pkl")
print(f"   • models/encoders.pkl")
print(f"   • models/feature_info.json")
print(f"   • models/feature_names.pkl")
print(f"   • models/model_comparison_report.json")
print(f"   • data/processed_credit_data.csv")
print(f"\n🎉 Training pipeline complete!")