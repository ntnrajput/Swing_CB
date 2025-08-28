# model/train_model.py

import sys
import os
import joblib
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier

from utils.logger import get_logger
from config import FEATURE_COLUMNS, MODEL_FILE

logger = get_logger(__name__)

def _safe_sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts dataframe chronologically if a 'date' column exists; otherwise returns as-is.
    Keeps compatibility with multi-ticker panels if already pre-sorted.
    """
    if 'date' in df.columns:
        try:
            # Ensure datetime
            if not np.issubdtype(df['date'].dtype, np.datetime64):
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Could not sort by date safely: {e}. Proceeding without explicit sort.")
    else:
        logger.warning("No 'date' column found; assuming input is already chronologically ordered.")
    return df

def _find_best_threshold(y_true, y_proba):
    """
    Choose decision threshold that maximizes F1 on the PR curve.
    Returns (best_threshold, best_f1, precision_at_best, recall_at_best).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds length = len(precision)-1; avoid division by zero
    f1_scores = []
    for p, r in zip(precision[:-1], recall[:-1]):
        if (p + r) == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * p * r / (p + r))
    if len(f1_scores) == 0:
        return 0.5, 0.0, 0.0, 0.0
    idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return best_threshold, float(f1_scores[idx]), float(precision[idx]), float(recall[idx])

def train_model(df_all, available_features=None, model_params=None):
    """
    Enhanced trading model with CatBoost, time-aware split, PR-focused tuning, and threshold optimization.
    (Function name, return structure, and key variable names preserved for compatibility.)
    """

    # -----------------------------
    # SECTION 0: Feature list checks
    # -----------------------------
    if available_features is None:
        try:
            from config import FEATURE_COLUMNS
            available_features = FEATURE_COLUMNS
        except ImportError:
            raise ValueError("available_features must be provided or FEATURE_COLUMNS must be available in config")

    missing_features = [col for col in available_features if col not in df_all.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")

    if 'target_hit' not in df_all.columns:
        raise ValueError("Target column 'target_hit' not found in dataframe")

    try:
        print("📊 Initial data shape:", df_all.shape)
        initial_rows = len(df_all)

        # -----------------------------
        # SECTION 1: Clean + (optional) chronological sort
        # -----------------------------
        num_cols = df_all.select_dtypes(include=[np.number]).columns
        df_all = df_all.copy()
        df_all[num_cols] = df_all[num_cols].replace([np.inf, -np.inf], np.nan)
        # Drop rows with NaN in features or target
        df_all = df_all.dropna(subset=available_features + ['target_hit']).reset_index(drop=True)
        # Time-aware ordering if possible
        df_all = _safe_sort_by_date(df_all)

        final_rows = len(df_all)
        print(f"✅ Dropped {initial_rows - final_rows} rows due to NaN/Inf values.")
        print("📊 Final data shape:", df_all.shape)

        # -----------------------------
        # SECTION 2: Feature-target split
        # -----------------------------
        X = df_all[available_features]
        y = df_all['target_hit'].astype(int)

        # -----------------------------
        # SECTION 3: Scaling (kept for compatibility with your pipeline)
        # NOTE: CatBoost does not require scaling; we keep it to avoid breaking downstream usage.
        # -----------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -----------------------------
        # SECTION 3.5: Calculate class weights early for feature selection
        # -----------------------------
        y_temp = y.astype(int)
        pos_temp = int(np.sum(y_temp))
        neg_temp = int(len(y_temp) - pos_temp)
        if pos_temp == 0:
            raise ValueError("No positive samples found. Check labeling logic or expand dataset.")
        
        # Calculate aggressive weight for feature selection
        base_weight_temp = neg_temp / pos_temp
        aggressive_weight = min(base_weight_temp * 2.0, 10.0)  # Cap at 10x to avoid instability
        logger.info(f"Early imbalance calc: pos={pos_temp}, neg={neg_temp}, aggressive_weight={aggressive_weight:.3f}")

        # -----------------------------
        # SECTION 4: Feature selection (made safe/optional)
        # We fit a lightweight CatBoost to compute importances, but avoid dropping too many features.
        # If selection yields < 5 features, we fallback to keeping all features.
        # -----------------------------

        catboost_fs = CatBoostClassifier(
            iterations=300,
            learning_rate=0.15,           # faster learning for feature selection
            depth=3,                      # shallower for stability
            colsample_bylevel=0.8,
            random_state=42,
            class_weights=[1, aggressive_weight],  # use aggressive weighting
            eval_metric="AUC",
            verbose=False,
            thread_count=-1,
            bootstrap_type='Bayesian',    # better for small positive class
            bagging_temperature=0.5
        )
        catboost_fs.fit(X_scaled, y)

        # Try median threshold; if too aggressive, keep all features
        selector = SelectFromModel(catboost_fs, threshold="median", prefit=True)
        X_selected = selector.transform(X_scaled)
        selected_mask = selector.get_support()
        selected_features = [f for f, keep in zip(available_features, selected_mask) if keep]

        if X_selected.shape[1] < 5:
            # Fallback: keep all features
            selector = SelectFromModel(catboost_fs, threshold=-np.inf, prefit=True)  # keep all
            X_selected = selector.transform(X_scaled)
            selected_features = available_features.copy()
            logger.info("Feature selection too aggressive. Falling back to keeping all features.")

        print(f"🔍 Selected Top {len(selected_features)} Features: {selected_features}")

        # -----------------------------
        # SECTION 5: Time-aware split (NO RANDOM SPLIT)
        # -----------------------------
        split_idx = int(len(X_selected) * 0.8)
        X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        # -----------------------------
        # SECTION 6: Imbalance handling via explicit class weights (ENHANCED)
        # -----------------------------
        pos = int(np.sum(y_train))
        neg = int(len(y_train) - pos)
        if pos == 0:
            raise ValueError("No positive samples in training set after split. Check labeling logic or expand dataset.")
        
        # More aggressive class weighting for extreme imbalance (recalculate for train set)
        base_weight = neg / pos
        aggressive_weight_train = min(base_weight * 2.0, 10.0)  # Cap at 10x to avoid instability
        logger.info(f"Train imbalance: pos={pos}, neg={neg}, base_weight={base_weight:.3f}, using_weight={aggressive_weight_train:.3f}")

        # -----------------------------
        # SECTION 7: Hyperparameter tuning (RandomizedSearchCV, PR-focused)
        # -----------------------------
        if model_params is None:
            # Focused parameters for extreme imbalance (removed subsample due to Bayesian bootstrap conflict)
            param_dist = {
                'iterations': [800, 1500, 2000],      # more iterations for convergence
                'depth': [3, 4, 5],                   # shallower trees for stability  
                'learning_rate': [0.01, 0.03, 0.05], # slower learning
                'colsample_bylevel': [0.8, 0.9],      # only column sampling, no row sampling
                'min_data_in_leaf': [5, 10, 20],      # require more samples per leaf
                'reg_lambda': [5, 10, 15],            # strong regularization
                'bootstrap_type': ['Bayesian'],        # better for imbalanced data
                'bagging_temperature': [0.3, 0.5, 0.8],
                'border_count': [64, 128]
            }
            search_cv = RandomizedSearchCV
            search_params = {
                'param_distributions': param_dist,
                'n_iter': 40,                      # balanced search
                'cv': 3,                           # faster with extreme imbalance  
                'scoring': 'roc_auc',              
                'n_jobs': -1,
                'verbose': 1,
                'random_state': 42,
                'return_train_score': True         # monitor overfitting
            }
        else:
            # If user passed a grid, still use PR metric
            search_cv = GridSearchCV
            search_params = {
                'param_grid': model_params,
                'cv': 5,
                'scoring': 'roc_auc',
                'n_jobs': -1,
                'verbose': 1
            }

        print("🔍 Performing hyperparameter tuning with CatBoost (Extreme Imbalance Focus)...")

        catboost_model = CatBoostClassifier(
            random_state=42,
            class_weights=[1, aggressive_weight_train],   # use aggressive weighting for training
            verbose=False,
            thread_count=-1,
            early_stopping_rounds=200,             # use only early_stopping_rounds
            eval_metric='AUC',
            od_type='IncToDec',                     # more conservative overfitting detection
            bootstrap_type='Bayesian',              # better for extreme imbalance
            leaf_estimation_method='Gradient'       # more stable leaf estimation
        )

        # Use early stopping via fit params on the time-aware validation fold
        # (Grid/RandomizedSearchCV will pass these on each fold)
        search = search_cv(
            estimator=catboost_model,
            **search_params
        )

        # Fit with minimal fit_params to avoid compatibility issues
        fit_params = {
            'eval_set': [(X_test, y_test)],
            'verbose': False
        }
        
        search.fit(X_train, y_train, **fit_params)

        best_model = search.best_estimator_
        print(f"✅ Best parameters: {search.best_params_}")

        # -----------------------------
        # SECTION 8: Evaluation + Threshold tuning
        # -----------------------------
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        # Find best threshold by F1 on PR curve (more conservative)
        decision_threshold, best_f1, p_at_best, r_at_best = _find_best_threshold(y_test.values, y_pred_proba)
        
        # Also try a more conservative threshold for precision
        precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_test, y_pred_proba)
        # Find threshold where precision >= 0.3 (targeting higher precision)
        precision_target = 0.3
        conservative_threshold = 0.5
        for i, p in enumerate(precision_curve[:-1]):
            if p >= precision_target:
                conservative_threshold = float(thresholds_curve[i])
                break
        
        print(f"📊 F1-optimized threshold: {decision_threshold:.4f}")
        print(f"📊 Precision-focused threshold: {conservative_threshold:.4f}")
        
        # Use the more conservative threshold
        final_threshold = max(decision_threshold, conservative_threshold)
        y_pred = (y_pred_proba >= final_threshold).astype(int)
        
        # Recalculate metrics with final threshold
        final_precision = precision_score(y_test, y_pred) if np.sum(y_pred) > 0 else 0.0
        final_recall = recall_score(y_test, y_pred)
        final_f1 = f1_score(y_test, y_pred)

        # Summary metrics
        ap = average_precision_score(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        print("\n🎯 Enhanced Model Evaluation (Extreme Imbalance Focus):")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Average Precision (PR AUC): {ap:.4f}")
        print(f"Final Threshold: {final_threshold:.4f}")
        print(f"Final Metrics: Precision={final_precision:.4f} | Recall={final_recall:.4f} | F1={final_f1:.4f}")
        print("Confusion Matrix (at final threshold):")
        print(cm)

        # -----------------------------
        # SECTION 9: Feature importance (CatBoost feature importance)
        # -----------------------------
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n🔝 Top 15 Most Important Features:")
        print(feature_importance.head(15))

        # Plot feature importance
        try:
            plt.figure(figsize=(12, 10))
            sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
            plt.title('CatBoost Feature Importance (Top 20)')
            plt.tight_layout()
            plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Feature importance plot saved as 'feature_importance_plot.png'")
        except Exception as e:
            print(f"⚠️ Could not create feature importance plot: {e}")

        # Plot Precision-Recall curve
        try:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.tight_layout()
            plt.savefig("pr_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("📈 PR curve saved as 'pr_curve.png'")
        except Exception as e:
            print(f"⚠️ Could not create PR curve: {e}")

        # -----------------------------
        # SECTION 10: Additional metrics (kept for compatibility)
        # -----------------------------
        train_pred_proba = best_model.predict_proba(X_train)[:, 1]
        # Use final threshold for train to compute comparable accuracy
        train_pred = (train_pred_proba >= final_threshold).astype(int)
        train_accuracy = float(np.mean(train_pred == y_train))
        test_accuracy = float(np.mean(y_pred == y_test))
        overfitting_gap = train_accuracy - test_accuracy
        win_rate = test_accuracy * 100.0

        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting_gap': overfitting_gap,
            'win_rate': win_rate,
            'roc_auc': roc_auc,
            'average_precision': float(ap),
            'decision_threshold': float(final_threshold),
            'precision_at_threshold': float(final_precision),
            'recall_at_threshold': float(final_recall),
            'f1_at_threshold': float(final_f1),
            'confusion_matrix': cm.tolist()
        }

        logger.info(f"Model metrics: {metrics}")

        # -----------------------------
        # SECTION 11: Save pipeline (names preserved)
        # -----------------------------
        full_pipeline = Pipeline([
            ('scaler', scaler),     # kept for compatibility
            ('selector', selector),
            ('model', best_model)
        ])

        model_bundle = {
            'pipeline': full_pipeline,
            'feature_columns': selected_features,   # Features after selection
            'all_features': available_features,     # Full feature list before selection
            'metrics': metrics,
            'best_params': search.best_params_,
            'feature_importance': feature_importance
        }

        joblib.dump(model_bundle, MODEL_FILE)
        logger.info(f"💾 Model pipeline saved to {MODEL_FILE}")

        print("🚀 CatBoost enhanced model saved!")

        return model_bundle

    except Exception as e:
        logger.error(f"Failed to train enhanced model: {e}", exc_info=True)
        raise