import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


DATA_FILE = "analytics/titanic.csv"


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    df = pd.read_csv(DATA_FILE)

    print("=" * 60)
    print("LOADING CLEAN TITANIC DATA")
    print("=" * 60)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


# =========================================================
# CLASSIFICATION DATA PREPARATION
# =========================================================

def prepare_data(df):

    print("\n" + "=" * 60)
    print("PREPARING FEATURES AND TARGET")
    print("=" * 60)

    y = df["survived"]

    X = df.drop(
        columns=[
            "survived",
            "alive"
        ]
    )

    X = X.drop(
        columns=[
            "deck",
            "embark_town",
            "class",
            "who",
            "adult_male",
            "alone"
        ]
    )

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nTarget:")
    print("survived")

    print("\nTarget distribution:")
    print(y.value_counts())

    return X, y


# =========================================================
# CLASSIFICATION TRAIN / TEST SPLIT
# =========================================================

def split_data(X, y):

    print("\n" + "=" * 60)
    print("STRATIFIED TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    print("\nTraining target distribution:")
    print(
        y_train.value_counts(
            normalize=True
        )
    )

    print("\nTesting target distribution:")
    print(
        y_test.value_counts(
            normalize=True
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# =========================================================
# PREPROCESSING
# =========================================================

def build_preprocessor(X_train):

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["str"]
    ).columns.tolist()

    print("\n" + "=" * 60)
    print("PREPROCESSING SETUP")
    print("=" * 60)

    print("\nNumeric features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# =========================================================
# BASELINE CLASSIFICATION MODELS
# =========================================================

def build_models(preprocessor):

    models = {

        "Logistic Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42
                    )
                )
            ]
        ),

        "Decision Tree": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    DecisionTreeClassifier(
                        random_state=42
                    )
                )
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=42
                    )
                )
            ]
        )
    }

    return models


# =========================================================
# CLASS-WEIGHT BALANCED MODELS
# =========================================================

def build_balanced_models(preprocessor):

    models = {

        "Logistic Regression - Balanced": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                        class_weight="balanced"
                    )
                )
            ]
        ),

        "Decision Tree - Balanced": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    DecisionTreeClassifier(
                        random_state=42,
                        class_weight="balanced"
                    )
                )
            ]
        ),

        "Random Forest - Balanced": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=42,
                        class_weight="balanced"
                    )
                )
            ]
        )
    }

    return models


# =========================================================
# SMOTE MODELS
# =========================================================

def build_smote_models(preprocessor):

    models = {

        "Logistic Regression - SMOTE": ImbPipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=42
                    )
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42
                    )
                )
            ]
        ),

        "Decision Tree - SMOTE": ImbPipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=42
                    )
                ),
                (
                    "model",
                    DecisionTreeClassifier(
                        random_state=42
                    )
                )
            ]
        ),

        "Random Forest - SMOTE": ImbPipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=42
                    )
                ),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=42
                    )
                )
            ]
        )
    }

    return models


# =========================================================
# TRAIN CLASSIFICATION MODELS
# =========================================================

def train_models(
    models,
    X_train,
    y_train,
    section_name
):

    print("\n" + "=" * 60)
    print(section_name)
    print("=" * 60)

    trained_models = {}

    for name, pipeline in models.items():

        print(f"\nTraining: {name}")

        pipeline.fit(
            X_train,
            y_train
        )

        trained_models[name] = pipeline

        print(
            f"{name} training completed."
        )

    return trained_models


# =========================================================
# EVALUATE CLASSIFICATION MODELS
# =========================================================

def evaluate_models(
    trained_models,
    X_test,
    y_test,
    section_name
):

    print("\n" + "=" * 60)
    print(section_name)
    print("=" * 60)

    results = []

    for name, model in trained_models.items():

        y_pred = model.predict(
            X_test
        )

        y_probability = model.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            y_probability
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        print(
            f"\n{name}"
        )

        print("-" * 40)

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print(
            f"ROC AUC  : {auc:.4f}"
        )

        print("\nConfusion Matrix:")
        print(cm)

        results.append(
            {
                "model": name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": auc
            }
        )

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 60)
    print(f"{section_name} COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df


# =========================================================
# DECISION TREE VISUALIZATION
# =========================================================

def plot_decision_tree(
    trained_models
):

    print("\n" + "=" * 60)
    print("DECISION TREE VISUALIZATION")
    print("=" * 60)

    decision_tree_pipeline = trained_models[
        "Decision Tree"
    ]

    preprocessor = decision_tree_pipeline.named_steps[
        "preprocessor"
    ]

    decision_tree_model = decision_tree_pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    print(
        f"\nTotal transformed features: "
        f"{len(feature_names)}"
    )

    print(
        f"Decision tree depth: "
        f"{decision_tree_model.get_depth()}"
    )

    print(
        f"Decision tree leaves: "
        f"{decision_tree_model.get_n_leaves()}"
    )

    plt.figure(
        figsize=(24, 14)
    )

    plot_tree(
        decision_tree_model,
        feature_names=feature_names,
        class_names=[
            "Did not survive",
            "Survived"
        ],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8
    )

    plt.title(
        "Decision Tree Visualization "
        "(First 3 Levels)"
    )

    plt.tight_layout()

    plt.savefig(
        "analytics/decision_tree.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\nDecision tree visualization saved to:")
    print(
        "analytics/decision_tree.png"
    )


# =========================================================
# RANDOM FOREST GRIDSEARCHCV + OOB
# =========================================================

def tune_random_forest(
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("RANDOM FOREST GRIDSEARCHCV")
    print("=" * 60)

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    bootstrap=True,
                    oob_score=True
                )
            )
        ]
    )

    parameter_grid = {
        "model__n_estimators": [
            100,
            200,
            300
        ],
        "model__max_depth": [
            None,
            5,
            10,
            15
        ],
        "model__max_features": [
            "sqrt",
            "log2"
        ]
    }

    print("\nParameter grid:")
    print(parameter_grid)

    print("\nStarting GridSearchCV...")

    grid_search = GridSearchCV(
        estimator=random_forest_pipeline,
        param_grid=parameter_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_

    print("\n" + "-" * 60)
    print("GRIDSEARCH RESULTS")
    print("-" * 60)

    print("\nBest parameters:")
    print(
        grid_search.best_params_
    )

    print(
        f"\nBest cross-validation F1 score: "
        f"{grid_search.best_score_:.4f}"
    )

    tuned_random_forest = best_model.named_steps[
        "model"
    ]

    print(
        f"OOB score: "
        f"{tuned_random_forest.oob_score_:.4f}"
    )

    y_pred = best_model.predict(
        X_test
    )

    y_probability = best_model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n" + "-" * 60)
    print("TUNED RANDOM FOREST TEST RESULTS")
    print("-" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC AUC  : {auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    tuning_results = pd.DataFrame(
        [
            {
                "model": "Random Forest - GridSearchCV",
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": auc,
                "cv_f1": grid_search.best_score_,
                "oob_score": tuned_random_forest.oob_score_
            }
        ]
    )

    print("\n" + "=" * 60)
    print("GRIDSEARCH RANDOM FOREST SUMMARY")
    print("=" * 60)

    print(
        tuning_results.to_string(
            index=False
        )
    )

    tuning_results.to_csv(
        "analytics/random_forest_gridsearch_results.csv",
        index=False
    )

    print("\nGridSearch results saved to:")
    print(
        "analytics/random_forest_gridsearch_results.csv"
    )

    return (
        best_model,
        tuning_results
    )


# =========================================================
# REGRESSION DATA PREPARATION
# =========================================================

def prepare_regression_data(df):

    print("\n" + "=" * 60)
    print("PREPARING REGRESSION DATA")
    print("=" * 60)

    y = df["fare"]

    X = df.drop(
        columns=[
            "fare"
        ]
    )

    X = X.drop(
        columns=[
            "alive",
            "deck",
            "embark_town",
            "class",
            "who",
            "adult_male",
            "alone"
        ]
    )

    print("\nRegression feature columns:")
    print(X.columns.tolist())

    print("\nRegression target:")
    print("fare")

    print(
        f"\nRegression rows: {len(X)}"
    )

    return X, y


# =========================================================
# REGRESSION TRAIN / TEST SPLIT
# =========================================================

def split_regression_data(
    X,
    y
):

    print("\n" + "=" * 60)
    print("REGRESSION TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(
        f"Regression training rows: "
        f"{len(X_train)}"
    )

    print(
        f"Regression testing rows : "
        f"{len(X_test)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# =========================================================
# REGRESSION PREPROCESSOR
# =========================================================

def build_regression_preprocessor(
    X_train
):

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["str"]
    ).columns.tolist()

    print("\n" + "=" * 60)
    print("REGRESSION PREPROCESSING SETUP")
    print("=" * 60)

    print("\nRegression numeric features:")
    print(numeric_features)

    print("\nRegression categorical features:")
    print(categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# =========================================================
# TRAIN LINEAR REGRESSION
# =========================================================

def train_linear_regression(
    preprocessor,
    X_train,
    y_train
):

    print("\n" + "=" * 60)
    print("TRAINING LINEAR REGRESSION")
    print("=" * 60)

    regression_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )

    regression_pipeline.fit(
        X_train,
        y_train
    )

    print(
        "\nLinear Regression training completed."
    )

    return regression_pipeline


# =========================================================
# EVALUATE LINEAR REGRESSION
# =========================================================

def evaluate_linear_regression(
    regression_pipeline,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("LINEAR REGRESSION EVALUATION")
    print("=" * 60)

    y_pred = regression_pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    n = len(y_test)

    fitted_preprocessor = (
        regression_pipeline.named_steps[
            "preprocessor"
        ]
    )

    feature_names = (
        fitted_preprocessor.get_feature_names_out()
    )

    p = len(feature_names)

    if n > p + 1:

        adjusted_r2 = (
            1
            - (
                (1 - r2)
                * (n - 1)
                / (n - p - 1)
            )
        )

    else:

        adjusted_r2 = float("nan")

    print(
        f"\nMAE : {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )

    print(
        f"Adjusted R²: {adjusted_r2:.4f}"
    )

    print(
        f"\nTest observations (n): {n}"
    )

    print(
        f"Transformed predictors (p): {p}"
    )

    # Predicted vs actual plot

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.7
    )

    minimum_value = min(
        y_test.min(),
        y_pred.min()
    )

    maximum_value = max(
        y_test.max(),
        y_pred.max()
    )

    plt.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        linestyle="--"
    )

    plt.xlabel(
        "Actual Fare"
    )

    plt.ylabel(
        "Predicted Fare"
    )

    plt.title(
        "Linear Regression: Actual vs Predicted Fare"
    )

    plt.tight_layout()

    plt.savefig(
        "analytics/regression_actual_vs_predicted.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Residuals

    residuals = (
        y_test - y_pred
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.7
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel(
        "Predicted Fare"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Residual Plot - Linear Regression"
    )

    plt.tight_layout()

    plt.savefig(
        "analytics/regression_residuals.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Residual spread

    residual_analysis = pd.DataFrame(
        {
            "predicted_fare": y_pred,
            "residual": residuals
        }
    )

    residual_analysis["predicted_quartile"] = pd.qcut(
        residual_analysis["predicted_fare"],
        q=4,
        duplicates="drop"
    )

    residual_spread = (
        residual_analysis
        .groupby(
            "predicted_quartile",
            observed=False
        )["residual"]
        .std()
    )

    print(
        "\nResidual standard deviation by predicted-fare quartile:"
    )

    print(
        residual_spread
    )

    print(
        "\nResidual analysis conclusion:"
    )

    if len(residual_spread) >= 2:

        minimum_std = residual_spread.min()
        maximum_std = residual_spread.max()

        if minimum_std > 0:

            spread_ratio = (
                maximum_std / minimum_std
            )

        else:

            spread_ratio = float("inf")

        if spread_ratio >= 2:

            print(
                "The residual spread changes substantially "
                "across predicted-fare ranges, which suggests "
                "heteroscedasticity."
            )

        else:

            print(
                "The residual spread is relatively similar "
                "across predicted-fare ranges, so there is "
                "no strong visual indication of "
                "heteroscedasticity."
            )

    else:

        print(
            "There are not enough residual groups to assess "
            "heteroscedasticity."
        )

    regression_results = pd.DataFrame(
        [
            {
                "model": "Linear Regression",
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "adjusted_r2": adjusted_r2
            }
        ]
    )

    regression_results.to_csv(
        "analytics/linear_regression_results.csv",
        index=False
    )

    residual_analysis.to_csv(
        "analytics/regression_residual_analysis.csv",
        index=False
    )

    print("\nRegression metrics saved to:")
    print(
        "analytics/linear_regression_results.csv"
    )

    print("\nResidual analysis saved to:")
    print(
        "analytics/regression_residual_analysis.csv"
    )

    print("\nRegression plots saved to:")
    print(
        "analytics/regression_actual_vs_predicted.png"
    )

    print(
        "analytics/regression_residuals.png"
    )

    return regression_results


# =========================================================
# FINAL MODEL COMPARISON
# =========================================================

def create_final_comparison(
    baseline_results,
    tuned_rf_results,
    regression_results
):

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    # -----------------------------------------------------
    # Classification comparison
    # -----------------------------------------------------

    classification_results = baseline_results.copy()

    tuned_classification = tuned_rf_results[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ]
    ].copy()

    classification_results = pd.concat(
        [
            classification_results[
                [
                    "model",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc"
                ]
            ],
            tuned_classification
        ],
        ignore_index=True
    )

    classification_results.insert(
        0,
        "metric_group",
        "Classification"
    )

    print("\nCLASSIFICATION RESULTS")
    print("-" * 60)

    print(
        classification_results.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Regression comparison
    # -----------------------------------------------------

    regression_final = regression_results.copy()

    regression_final.insert(
        0,
        "metric_group",
        "Regression"
    )

    print("\nREGRESSION RESULTS")
    print("-" * 60)

    print(
        regression_final.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Combined final table
    # -----------------------------------------------------

    final_comparison = pd.concat(
        [
            classification_results,
            regression_final
        ],
        ignore_index=True,
        sort=False
    )

    final_comparison.to_csv(
        "analytics/final_model_comparison.csv",
        index=False
    )

    print("\nFinal comparison saved to:")
    print(
        "analytics/final_model_comparison.csv"
    )

    return final_comparison


# =========================================================
# SAVE AND RELOAD FINAL PIPELINES
# =========================================================

def save_reload_pipelines(
    best_random_forest,
    linear_regression_model,
    X_test,
    regression_X_test
):

    print("\n" + "=" * 60)
    print("JOBLIB PIPELINE SAVE / RELOAD")
    print("=" * 60)

    # -----------------------------------------------------
    # Save classification pipeline
    # -----------------------------------------------------

    classification_pipeline_file = (
        "analytics/final_random_forest_pipeline.joblib"
    )

    joblib.dump(
        best_random_forest,
        classification_pipeline_file
    )

    print(
        "\nClassification Pipeline saved to:"
    )

    print(
        classification_pipeline_file
    )

    # -----------------------------------------------------
    # Reload classification pipeline
    # -----------------------------------------------------

    loaded_classification_pipeline = joblib.load(
        classification_pipeline_file
    )

    classification_predictions = (
        loaded_classification_pipeline.predict(
            X_test.head(5)
        )
    )

    print(
        "\nReloaded classification Pipeline predictions:"
    )

    print(
        classification_predictions.tolist()
    )

    # -----------------------------------------------------
    # Save regression pipeline
    # -----------------------------------------------------

    regression_pipeline_file = (
        "analytics/linear_regression_pipeline.joblib"
    )

    joblib.dump(
        linear_regression_model,
        regression_pipeline_file
    )

    print(
        "\nRegression Pipeline saved to:"
    )

    print(
        regression_pipeline_file
    )

    # -----------------------------------------------------
    # Reload regression pipeline
    # -----------------------------------------------------

    loaded_regression_pipeline = joblib.load(
        regression_pipeline_file
    )

    regression_predictions = (
        loaded_regression_pipeline.predict(
            regression_X_test.head(5)
        )
    )

    print(
        "\nReloaded regression Pipeline predictions:"
    )

    print(
        regression_predictions.tolist()
    )

    print("\nJoblib save/reload demonstration completed.")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------

    df = load_data()

    # -----------------------------------------------------
    # 2. Classification preparation
    # -----------------------------------------------------

    X, y = prepare_data(
        df
    )

    # -----------------------------------------------------
    # 3. Classification split
    # -----------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    # -----------------------------------------------------
    # 4. Classification preprocessing
    # -----------------------------------------------------

    preprocessor = build_preprocessor(
        X_train
    )

    # -----------------------------------------------------
    # 5. Baseline classification
    # -----------------------------------------------------

    baseline_models = build_models(
        preprocessor
    )

    baseline_trained = train_models(
        baseline_models,
        X_train,
        y_train,
        "TRAINING BASELINE MODELS"
    )

    baseline_results = evaluate_models(
        baseline_trained,
        X_test,
        y_test,
        "BASELINE MODEL EVALUATION"
    )

    # -----------------------------------------------------
    # 6. Class-weight balanced
    # -----------------------------------------------------

    balanced_models = build_balanced_models(
        preprocessor
    )

    balanced_trained = train_models(
        balanced_models,
        X_train,
        y_train,
        "TRAINING CLASS-WEIGHT BALANCED MODELS"
    )

    balanced_results = evaluate_models(
        balanced_trained,
        X_test,
        y_test,
        "CLASS-WEIGHT BALANCED EVALUATION"
    )

    # -----------------------------------------------------
    # 7. SMOTE
    # -----------------------------------------------------

    smote_models = build_smote_models(
        preprocessor
    )

    smote_trained = train_models(
        smote_models,
        X_train,
        y_train,
        "TRAINING SMOTE MODELS"
    )

    smote_results = evaluate_models(
        smote_trained,
        X_test,
        y_test,
        "SMOTE EVALUATION"
    )

    # -----------------------------------------------------
    # 8. Class imbalance comparison
    # -----------------------------------------------------

    all_results = pd.concat(
        [
            baseline_results.assign(
                approach="Baseline"
            ),
            balanced_results.assign(
                approach="Class Weight Balanced"
            ),
            smote_results.assign(
                approach="SMOTE"
            )
        ],
        ignore_index=True
    )

    print("\n" + "=" * 60)
    print("CLASS IMBALANCE COMPARISON")
    print("=" * 60)

    print(
        all_results[
            [
                "approach",
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc"
            ]
        ].to_string(
            index=False
        )
    )

    all_results.to_csv(
        "analytics/class_imbalance_comparison.csv",
        index=False
    )

    print("\nComparison saved to:")
    print(
        "analytics/class_imbalance_comparison.csv"
    )

    # -----------------------------------------------------
    # 9. Decision Tree visualization
    # -----------------------------------------------------

    plot_decision_tree(
        baseline_trained
    )

    # -----------------------------------------------------
    # 10. Random Forest GridSearchCV + OOB
    # -----------------------------------------------------

    (
        best_random_forest,
        gridsearch_results
    ) = tune_random_forest(
        preprocessor,
        X_train,
        y_train,
        X_test,
        y_test
    )

    # -----------------------------------------------------
    # 11. Regression preparation
    # -----------------------------------------------------

    regression_X, regression_y = (
        prepare_regression_data(
            df
        )
    )

    # -----------------------------------------------------
    # 12. Regression split
    # -----------------------------------------------------

    (
        regression_X_train,
        regression_X_test,
        regression_y_train,
        regression_y_test
    ) = split_regression_data(
        regression_X,
        regression_y
    )

    # -----------------------------------------------------
    # 13. Regression preprocessing
    # -----------------------------------------------------

    regression_preprocessor = (
        build_regression_preprocessor(
            regression_X_train
        )
    )

    # -----------------------------------------------------
    # 14. Train Linear Regression
    # -----------------------------------------------------

    linear_regression_model = (
        train_linear_regression(
            regression_preprocessor,
            regression_X_train,
            regression_y_train
        )
    )

    # -----------------------------------------------------
    # 15. Evaluate Linear Regression
    # -----------------------------------------------------

    regression_results = (
        evaluate_linear_regression(
            linear_regression_model,
            regression_X_test,
            regression_y_test
        )
    )

    # -----------------------------------------------------
    # 16. Final comparison
    # -----------------------------------------------------

    final_comparison = create_final_comparison(
        baseline_results,
        gridsearch_results,
        regression_results
    )

    # -----------------------------------------------------
    # 17. Save and reload pipelines
    # -----------------------------------------------------

    save_reload_pipelines(
        best_random_forest,
        linear_regression_model,
        X_test,
        regression_X_test
    )

    # -----------------------------------------------------
    # 18. Completion message
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("ANALYTICS MODULE COMPLETED")
    print("=" * 60)