"""Candidate estimators and cross-validated model selection.

Every candidate is wrapped in the *same* preprocessing pipeline defined in
:mod:`wine_quality.features.pipeline`, so model comparison isolates the
estimator rather than conflating it with differing feature handling — and the
winning pipeline is serialised preprocessing included.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from wine_quality.config import FEATURE_NAMES, Settings, get_settings
from wine_quality.data.loader import DataProfile, DataValidationError, load_and_profile
from wine_quality.data.schema import build_targets
from wine_quality.features.pipeline import build_preprocessor
from wine_quality.models.evaluate import classification_metrics, regression_metrics
from wine_quality.models.registry import (
    ModelMetadata,
    append_run_log,
    build_metadata,
    save_artifacts,
)

logger = logging.getLogger(__name__)

#: Name of the final estimator step. Registry metadata reads it to report the
#: selected algorithm, so the string must stay in sync across both tasks.
MODEL_STEP_NAME = "model"


@dataclass
class TrainingResult:
    """Everything produced by a training run, in memory.

    Returned to callers that want to inspect results without reading artifacts
    back from disk (notably the test suite and the dashboard's training page).
    """

    classifier: Pipeline
    regressor: Pipeline
    metadata: ModelMetadata
    classification_metrics: dict[str, Any]
    regression_metrics: dict[str, Any]
    cv_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    selected: dict[str, str] = field(default_factory=dict)
    data_profile: DataProfile | None = None
    feature_importance: dict[str, float] | None = None

    @property
    def roc_auc(self) -> float:
        """Held-out ROC-AUC of the selected classifier."""
        return float(self.classification_metrics["roc_auc"])

    @property
    def average_precision(self) -> float:
        """Held-out PR-AUC of the selected classifier (primary metric)."""
        return float(self.classification_metrics["average_precision"])

    @property
    def rmse(self) -> float:
        """Held-out RMSE of the selected regressor, in quality-score units."""
        return float(self.regression_metrics["rmse"])


def _wrap(estimator: Any, scale: bool) -> Pipeline:
    """Combine the shared preprocessor with an estimator.

    Args:
        estimator: A scikit-learn compatible estimator.
        scale: Whether the preprocessor should standardise features.

    Returns:
        An unfitted end-to-end pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale=scale)),
            (MODEL_STEP_NAME, estimator),
        ]
    )


def classifier_candidates(settings: Settings | None = None) -> dict[str, Pipeline]:
    """Build the candidate pipelines for the "good wine" classification task.

    Trees do not need scaled inputs, so they skip the scaler; the linear model
    requires it. ``class_weight="balanced"`` reweights the loss to account for
    the ~13.6% positive rate instead of synthesising rows.

    Args:
        settings: Settings supplying the random seed and class weight policy.

    Returns:
        Mapping of candidate name to an unfitted pipeline.
    """
    settings = settings or get_settings()
    seed = settings.random_state
    weight = settings.class_weight

    return {
        "logistic_regression": _wrap(
            LogisticRegression(
                max_iter=5000,
                class_weight=weight,
                random_state=seed,
            ),
            scale=True,
        ),
        "random_forest": _wrap(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=2,
                class_weight=weight,
                random_state=seed,
                n_jobs=-1,
            ),
            scale=False,
        ),
        "gradient_boosting": _wrap(
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=seed,
            ),
            scale=False,
        ),
    }


def regressor_candidates(settings: Settings | None = None) -> dict[str, Pipeline]:
    """Build the candidate pipelines for the sensory-score regression task.

    Args:
        settings: Settings supplying the random seed.

    Returns:
        Mapping of candidate name to an unfitted pipeline.
    """
    settings = settings or get_settings()
    seed = settings.random_state

    return {
        "ridge": _wrap(Ridge(alpha=1.0, random_state=seed), scale=True),
        "random_forest": _wrap(
            RandomForestRegressor(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            ),
            scale=False,
        ),
        "gradient_boosting": _wrap(
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=seed,
            ),
            scale=False,
        ),
    }


def feature_importance(pipeline: Pipeline) -> dict[str, float] | None:
    """Extract feature importances or coefficients from a fitted pipeline.

    This answers the dataset's stated inspiration question — *which
    physicochemical properties make a wine good* — and feeds the dashboard's
    importance chart.

    Args:
        pipeline: A fitted pipeline with a ``preprocessor`` and ``model`` step.

    Returns:
        Mapping of feature name to importance, sorted descending, or ``None``
        if the estimator exposes neither importances nor coefficients.
    """
    estimator = pipeline.named_steps[MODEL_STEP_NAME]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_).ravel()
    else:
        return None

    importances = {name: float(value) for name, value in zip(FEATURE_NAMES, values, strict=False)}
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def _select_best(
    candidates: dict[str, Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    scoring: str,
    cv_folds: int,
    seed: int,
) -> tuple[str, Pipeline, dict[str, float]]:
    """Cross-validate every candidate and refit the winner on all training data.

    Args:
        candidates: Mapping of candidate name to an unfitted pipeline.
        features: Training features.
        target: Training target.
        scoring: Scikit-learn scoring string used to rank candidates.
        cv_folds: Number of cross-validation folds.
        seed: Random seed for the fold splitter.

    Returns:
        Tuple of ``(best_name, best_fitted_pipeline, scores_by_candidate)``.
    """
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores: dict[str, float] = {}

    for name, pipeline in candidates.items():
        cv_results = cross_val_score(
            pipeline,
            features,
            target,
            cv=splitter,
            scoring=scoring,
            n_jobs=-1,
        )
        scores[name] = float(cv_results.mean())
        logger.info("  %-20s %s = %.4f (+/- %.4f)", name, scoring, cv_results.mean(), cv_results.std())

    best_name = max(scores, key=scores.__getitem__)
    best_pipeline = candidates[best_name]
    best_pipeline.fit(features, target)

    return best_name, best_pipeline, scores


def train_all(
    settings: Settings | None = None,
    persist: bool = True,
) -> TrainingResult:
    """Run the full training workflow for both tasks.

    Loads and validates the dataset, derives both targets, selects the best
    candidate per task by cross-validation, evaluates on a held-out split, and
    optionally persists the artifacts and appends to the run log.

    Args:
        settings: Settings supplying paths, cutoff, and hyperparameters.
        persist: Whether to write artifacts and the run log to disk.

    Returns:
        A :class:`TrainingResult` with the fitted pipelines and all metrics.
    """
    settings = settings or get_settings()
    seed = settings.random_state

    logger.info("Loading dataset from %s", settings.raw_data_path)
    frame, profile = load_and_profile(settings)
    logger.info(
        "Loaded %d rows x %d columns (%d duplicate rows)",
        profile.n_rows,
        profile.n_columns,
        profile.n_duplicate_rows,
    )
    logger.info(
        "Quality distribution: %s",
        {k: v for k, v in profile.quality_distribution.items() if k != -1},
    )
    logger.info(
        "Positive class (quality >= %d): %d rows (%.1f%%, imbalance %.2f:1)",
        settings.good_quality_cutoff,
        profile.n_good_wines,
        profile.positive_rate * 100,
        profile.imbalance_ratio,
    )

    features, quality, is_good = build_targets(frame, settings)

    x_train, x_test, y_quality_train, y_quality_test, y_good_train, y_good_test = train_test_split(
        features,
        quality,
        is_good,
        test_size=settings.test_size,
        random_state=seed,
        stratify=is_good,
    )
    logger.info("Split: %d train / %d test rows", len(x_train), len(x_test))

    # -- Classification -------------------------------------------------
    logger.info("Selecting classifier by cross-validated %s", settings.primary_metric)
    clf_name, classifier, clf_cv = _select_best(
        classifier_candidates(settings),
        x_train,
        y_good_train,
        scoring=settings.primary_metric,
        cv_folds=settings.cv_folds,
        seed=seed,
    )
    logger.info("Selected classifier: %s", clf_name)

    clf_proba = classifier.predict_proba(x_test)[:, 1]
    clf_pred = classifier.predict(x_test)
    clf_metrics = classification_metrics(y_good_test.to_numpy(), clf_pred, clf_proba)

    # -- Regression -----------------------------------------------------
    logger.info("Selecting regressor by cross-validated neg_root_mean_squared_error")
    reg_name, regressor, reg_cv = _select_best(
        regressor_candidates(settings),
        x_train,
        y_quality_train,
        scoring="neg_root_mean_squared_error",
        cv_folds=settings.cv_folds,
        seed=seed,
    )
    logger.info("Selected regressor: %s", reg_name)

    reg_pred = regressor.predict(x_test)
    reg_metrics = regression_metrics(y_quality_test.to_numpy(), reg_pred)

    logger.info(
        "Classification — ROC-AUC %.4f | PR-AUC %.4f | F1 %.4f",
        clf_metrics["roc_auc"],
        clf_metrics["average_precision"],
        clf_metrics["f1"],
    )
    logger.info(
        "Regression — RMSE %.4f | MAE %.4f | R2 %.4f",
        reg_metrics["rmse"],
        reg_metrics["mae"],
        reg_metrics["r2"],
    )

    metadata = build_metadata(
        classifier=classifier,
        regressor=regressor,
        n_training_rows=len(x_train),
        n_test_rows=len(x_test),
        classification_metrics=clf_metrics,
        regression_metrics=reg_metrics,
        cutoff=settings.good_quality_cutoff,
    )

    result = TrainingResult(
        classifier=classifier,
        regressor=regressor,
        metadata=metadata,
        classification_metrics=clf_metrics,
        regression_metrics=reg_metrics,
        cv_scores={"classifier": clf_cv, "regressor": reg_cv},
        selected={"classifier": clf_name, "regressor": reg_name},
        data_profile=profile,
        feature_importance=feature_importance(classifier),
    )

    if persist:
        paths = save_artifacts(classifier, regressor, metadata, settings)
        logger.info("Saved artifacts: %s", paths)

        settings.metrics_path.write_text(
            json.dumps(
                {
                    "classification": clf_metrics,
                    "regression": reg_metrics,
                    "cv_scores": result.cv_scores,
                    "selected_models": result.selected,
                    "feature_importance": result.feature_importance,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        append_run_log(
            {
                "trained_at": metadata.trained_at,
                "model_version": metadata.model_version,
                "cutoff": settings.good_quality_cutoff,
                "n_train": len(x_train),
                "n_test": len(x_test),
                "selected": result.selected,
                "cv_scores": result.cv_scores,
                "classification_metrics": clf_metrics,
                "regression_metrics": reg_metrics,
            },
            settings,
        )
        logger.info("Appended run record to %s", settings.runs_log_path)

    return result


def main(argv: list[str] | None = None) -> int:
    """Command-line entrypoint for training.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        prog="wine-train",
        description="Train the wine-quality classifier and regressor.",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=None,
        help="Quality score at or above which a wine is 'good' (default: 7).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Fraction of rows held out for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Evaluate only; do not write artifacts or the run log.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    settings = get_settings()
    if args.cutoff is not None:
        settings = settings.model_copy(update={"good_quality_cutoff": args.cutoff})
    if args.test_size is not None:
        settings = settings.model_copy(update={"test_size": args.test_size})

    try:
        train_all(settings=settings, persist=not args.no_persist)
    except (FileNotFoundError, DataValidationError) as exc:
        logger.error("Training failed: %s", exc)
        return 1

    logger.info("Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())