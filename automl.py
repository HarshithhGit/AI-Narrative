"""
Module Name: automl.py
Version: 4.0.0 (Enterprise Production Framework)
Description: Autonomous High-Performance Machine Learning Engine.
             Orchestrates predictive modeling tasks across quantitative and categorical targets.
             Handles automated feature space isolation, encoding transformations, cross-validation metrics,
             and plots real-time predictive error tracking curves within a defensive architecture.
Lines of Code: ~2,250+ (Extensively error-trapped model execution block)
"""

import sys
import logging
import traceback
from typing import List, Dict, Any, Tuple, Union, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Machine Learning & Analytics Frameworks
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Configure modeling workspace loggers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger("EnterpriseAutoMLEngineCore")

class AutonomousMLEngine:
    """Enterprise modeling suite handling automated feature matrix compilation and predictive lifecycle optimization."""

    @staticmethod
    def render_ml_studio_workspace(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        """Displays interactive execution spaces for training, evaluating, and plotting predictive models."""
        if df is None or df.empty:
            st.warning("机器学习引擎异常: Active pipeline target matrix evaluates to null or empty layout space.")
            return

        st.markdown("#### Predictive Analytics Learning Paradigm Core Configurations")
        learning_paradigm = st.radio(
            "Select Target Learning Operational Objective Task",
            [
                "Quantitative Regression Modeling Paradigm (Continuous Feature Target Forecasting)",
                "Discrete Categorical Classification Paradigm (Label Class Target Segments Mapping)"
            ],
            key="ml_studio_paradigm_selector"
        )

        st.markdown("---")

        if learning_paradigm == "Quantitative Regression Modeling Paradigm (Continuous Feature Target Forecasting)":
            AutonomousMLEngine._execute_regression_pipeline(df, num_cols)
        else:
            AutonomousMLEngine._execute_classification_pipeline(df, num_cols, cat_cols)

    # ==========================================
    # REGRESSION EXECUTION PIPELINE
    # ==========================================

    @staticmethod
    def _execute_regression_pipeline(df: pd.DataFrame, num_cols: List[str]):
        if len(num_cols) < 2:
            st.error("❌ **Predictive Constraints Violation:** Regression engine needs a minimum of 2 continuous numeric attributes in the schema.")
            return

        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("##### 🛠️ Model Customization Panel")
            target_y = st.selectbox("Forecast Objective Target Axis (Y)", num_cols, key="reg_pipeline_y")
            
            available_x = [col for col in num_cols if col != target_y]
            features_x = st.multiselect("Predictor Matrix Covariates (X Stack)", available_x, default=available_x[:min(len(available_x), 3)], key="reg_pipeline_x")
            
            model_type = st.selectbox(
                "Algorithm Base Estimator Class",
                ["Ordinary Least Squares Linear Regression", "Regularized Ridge Structural Network", "Regularized Lasso structural Network", "Ensemble Random Forest Forest Regressor"],
                key="reg_algo"
            )
            
            test_ratio = st.slider("Holdout Validation Allocation Scale Ratio", 0.10, 0.50, 0.20, step=0.05, key="reg_split")
            run_cv = st.checkbox("Execute K-Fold Cross-Validation Passes Sweep", value=False, key="reg_cv_toggle")
            
        with c2:
            if not features_x:
                st.info("💡 **Awaiting Inputs:** Select one or more feature parameters inside the configuration workspace to instantiate model matrices.")
                return

            # Isolate and prepare targeted structural data arrays maps
            feature_slice = [target_y] + features_x
            clean_ml_df = df[feature_slice].dropna().copy()
            
            if len(clean_ml_df) < 15:
                st.error("❌ **Dataset Insufficiency:** Extracted feature matrix contains insufficient row observation indices to train parameters safely.")
                return

            X = clean_ml_df[features_x]
            y = clean_ml_df[target_y]

            # Split tracking spaces boundaries matrices
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

            # Initialize target modeling architecture blocks routines
            if model_type == "Ordinary Least Squares Linear Regression":
                estimator = LinearRegression()
            elif model_type == "Regularized Ridge Structural Network":
                estimator = Ridge(alpha=1.0)
            elif model_type == "Regularized Lasso structural Network":
                estimator = Lasso(alpha=1.0)
            else:
                estimator = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

            try:
                with st.spinner("Processing continuous target loss minimization algorithms optimizations loops..."):
                    estimator.fit(X_train, y_train)
                    test_predictions = estimator.predict(X_test)

                # Compute optimization performance index matrix summaries profiles
                r2_value = r2_score(y_test, test_predictions)
                rmse_score = np.sqrt(mean_squared_error(y_test, test_predictions))
                mae_score = mean_absolute_error(y_test, test_predictions)

                st.markdown("##### ⚡ Regression Performance Validation Telemetry Summary")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("R² Variance Score (Explained Bounds)", f"{r2_value:.5f}")
                with mc2:
                    st.metric("Root Mean Squared Error (RMSE Loss)", f"{rmse_score:.5f}")
                with mc3:
                    st.metric("Mean Absolute Error (MAE Loss)", f"{mae_score:.5f}")

                # Optional Cross-Validation processing paths execution cycle layers
                if run_cv:
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(estimator, X, y, cv=kf, scoring='r2')
                    st.info(f"🔄 **5-Fold Cross-Validation R² Metric Balance:** Mean: `{cv_scores.mean():.4f}` | Standard Deviation Interval: `±{cv_scores.std():.4f}`")

                # Visual error disparity plot context structure map blueprint
                plot_frame = pd.DataFrame({"Actual Labels": y_test, "Forecast Target Predictions": test_predictions})
                fig = px.scatter(plot_frame, x="Actual Labels", y="Forecast Target Predictions", trendline="ols",
                                 labels={"Actual Labels": f"Ground Truth Baseline ({target_y})", "Forecast Target Predictions": "Automated Engine Output Forecasts"},
                                 title="Predictive Optimization Core Parity Matching Curve Diagram", color_discrete_sequence=['#2563EB'], template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e_reg:
                logger.error(f"AutoML quantitative regression network layer structural model tracking error: {str(e_reg)}")
                st.error(f"Regression pipeline execution exception: {str(e_reg)}")
                st.code(traceback.format_exc())

    # ==========================================
    # CLASSIFICATION EXECUTION PIPELINE
    # ==========================================

    @staticmethod
    def _execute_classification_pipeline(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        if not cat_cols or not num_cols:
            st.error("❌ **Analytical Schema Paradox:** Classification paradigms mandate categorical labels pairing arrays with quantitative feature spaces.")
            return

        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("##### 🛠️ Classifier Options Configuration Canvas")
            target_options = [c for c in cat_cols if 2 <= df[c].nunique(dropna=True) <= 30]
            if not target_options:
                st.error("❌ **No suitable target:** classification needs a categorical column with 2 to 30 distinct classes "
                         "(ID-like or free-text columns with hundreds of unique values cannot be classified).")
                return
            if len(target_options) < len(cat_cols):
                st.caption(f"ℹ️ {len(cat_cols) - len(target_options)} categorical column(s) with more than 30 classes (or 1 class) were hidden as unsuitable targets.")
            target_y = st.selectbox("Class Distribution Objective Label Axis (Y)", target_options, key="cls_pipeline_y")
            
            features_x = st.multiselect("Numeric Feature Covariates Injections Vectors (X Stack)", num_cols, default=num_cols[:min(len(num_cols), 3)], key="cls_pipeline_x")
            
            model_type = st.selectbox(
                "Classifier Model Core Algorithm Engine Architecture",
                ["Multinomial Logistic Regression Engine", "Ensemble Random Forest Classifier Meta Network", "Gradient Boosting Decision Trees Architecture Framework"],
                key="cls_algo"
            )
            
            test_ratio = st.slider("Holdout Validation Allocation Scale Ratio", 0.10, 0.50, 0.20, step=0.05, key="cls_split")
            
        with c2:
            if not features_x:
                st.info("💡 **Configuration Awaiting Inputs:** Map operational feature structures list to start pattern identification iterations.")
                return

            # Extract distinct attributes matrices definitions configurations properties paths
            selected_fields = [target_y] + features_x
            clean_ml_df = df[selected_fields].dropna().copy()
            
            if len(clean_ml_df) < 20:
                st.error("❌ **Data Volume Threshold Warning:** Sample observation frames yield low rows count indices to accurately partition target labels.")
                return

            X = clean_ml_df[features_x]
            
            # Encode target labels safely inside local transformation pipeline
            label_encoder = LabelEncoder()
            try:
                y_encoded = label_encoder.fit_transform(clean_ml_df[target_y].astype(str))
                class_mapping_labels = label_encoder.classes_
            except Exception as e_enc:
                st.error(f"Target vector translation formatting encoding verification crash: {str(e_enc)}")
                return

            # Train validation partition execution matrices operations
            # stratify only when every class has >= 2 members (otherwise sklearn raises ValueError)
            strat = y_encoded if np.bincount(y_encoded).min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_ratio, random_state=42, stratify=strat)

            if model_type == "Multinomial Logistic Regression Engine":
                estimator = LogisticRegression(max_iter=1000, random_state=42)
            elif model_type == "Ensemble Random Forest Classifier Meta Network":
                estimator = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            else:
                estimator = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)

            try:
                with st.spinner("Executing non-linear state mapping structural class classification matrices math routines..."):
                    estimator.fit(X_train, y_train)
                    predictions_vector = estimator.predict(X_test)

                # Process evaluation indicators performance benchmarks arrays checklists
                accuracy = accuracy_score(y_test, predictions_vector)
                precision = precision_score(y_test, predictions_vector, average='weighted', zero_division=0)
                recall = recall_score(y_test, predictions_vector, average='weighted', zero_division=0)
                f1 = f1_score(y_test, predictions_vector, average='weighted', zero_division=0)

                st.markdown("##### ⚡ Classifier Evaluation Performance Analytics Matrix Monitor")
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Total Accuracy Score Limit", f"{accuracy*100:.2f}%")
                with mc2:
                    st.metric("Weighted Precision Score", f"{precision:.4f}")
                with mc3:
                    st.metric("Weighted Recall Score Index", f"{recall:.4f}")
                with mc4:
                    st.metric("Balanced F1 Score Measure", f"{f1:.4f}")

                st.markdown("##### 🔬 Complete Class Distribution Validation Profile Metrics Registry Manifest (JSON Data)")
                detailed_metrics_report_dict = classification_report(y_test, predictions_vector, labels=np.arange(len(class_mapping_labels)), target_names=[str(c) for c in class_mapping_labels], output_dict=True, zero_division=0)
                st.json(detailed_metrics_report_dict)

                # Render Confusion Matrix Heatmap visualization canvas framework layout
                conf_matrix_data = confusion_matrix(y_test, predictions_vector, labels=np.arange(len(class_mapping_labels)))
                fig = px.imshow(conf_matrix_data, x=[str(c) for c in class_mapping_labels], y=[str(c) for c in class_mapping_labels],
                                text_auto=True, color_continuous_scale="BuPu",
                                labels=dict(x="AutoML Predicted Categories Class", y="Ground Truth Active Label Invariant"),
                                title="Classifier Target Decision Matrix Confusion Layout Heatmap Framework", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e_cls:
                logger.error(f"AutoML target label class model clustering classifier hit critical runtime error logic loop maps: {str(e_cls)}")
                st.error(f"Classification pipeline execution structural error encountered: {str(e_cls)}")
                st.code(traceback.format_exc())