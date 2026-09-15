"""Tests for model training, evaluation, and artifact persistence.

The end-to-end training test is marked ``slow`` because it cross-validates six
candidate pipelines. Run ``pytest -m "not slow"`` for a fast feedback loop.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wine_quality.config import FEATURE_NAMES
from wine_quality.models.evaluate import (
    classification_metrics,
    metrics_to_display,
    regression_metrics,
)
from wine_quality.models.registry import (
    ArtifactError,
    ModelMetadata,
    append_run_log,
    artifacts_exist,
    load_artifacts,
    read_run_log,
    save_artifacts,
)
from wine_quality.models.train import (
    classifier_candidates,
    feature_importance,
    regressor_candidates,
    train_all,
)


class TestCandidates:
    """Candidate pipelines must be well-formed and share the preprocessing step."""

    def test_classifier_candidates_are_pipelines(self) -> None:
        candidates = classifier_candidates()
        assert set(candidates) == {"logistic_regression", "random_forest", "gradient_boosting"}
        for pipeline in candidates.values():
            assert list(pipeline.named_steps) == ["preprocessor", "model"]

    def test_regressor_candidates_are_pipelines(self) -> None:
        candidates = regressor_candidates()
        assert set(candidates) == {"ridge", "random_forest", "gradient_boosting"}
        for pipeline in candidates.values():
            assert list(pipeline.named_steps) == ["preprocessor", "model"]

    def test_linear_models_use_scaling(self) -> None:
        """Linear models need standardised inputs; trees do not."""

        def numeric_steps(pipeline) -> list[str]:
            # `.transformers` works on an unfitted ColumnTransformer, whereas
            # `.named_transformers_` only exists after fitting.
            for name, transformer, _columns in pipeline.named_steps["preprocessor"].transformers:
                if name == "numeric":
                    return list(transformer.named_steps)
            raise AssertionError("no numeric branch found in the preprocessor")

        assert "scaler" in numeric_steps(classifier_candidates()["logistic_regression"])
        assert "scaler" not in numeric_steps(classifier_candidates()["random_forest"])
        assert "scaler" in numeric_steps(regressor_candidates()["ridge"])
        assert "scaler" not in numeric_steps(regressor_candidates()["gradient_boosting"])

    def test_classifier_uses_balanced_weights(self) -> None:
        """The imbalance mitigation must actually be wired in."""
        estimator = classifier_candidates()["logistic_regression"].named_steps["model"]
        assert estimator.class_weight == "balanced"

    def test_candidates_are_independent_instances(self) -> None:
        """Mutating one candidate must not affect another."""
        first = classifier_candidates()
        second = classifier_candidates()
        assert first["random_forest"] is not second["random_forest"]


class TestEvaluationMetrics:
    """Metric functions must be correct on hand-checkable inputs."""

    def test_perfect_classifier_scores_one(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        metrics = classification_metrics(y_true, y_true, y_true.astype(float))
        assert metrics["roc_auc"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["average_precision"] == 1.0

    def test_confusion_matrix_counts_are_correct(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = classification_metrics(y_true, y_pred, np.array([0.1, 0.6, 0.4, 0.9]))
        assert metrics["true_negatives"] == 1
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1
        assert metrics["true_positives"] == 1
        assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]

    def test_all_negative_predictions_give_zero_recall(self) -> None:
        """The trivial 'never good' model — the failure mode the metrics must expose.

        Probabilities are deliberately anti-correlated with the truth, so the
        ranking is poor as well as the thresholded prediction.
        """
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.zeros(4, dtype=int)
        y_proba = np.array([0.9, 0.8, 0.3, 0.2])

        metrics = classification_metrics(y_true, y_pred, y_proba)

        assert metrics["recall"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["f1"] == 0.0
        # Accuracy looks respectable while the model is useless...
        assert metrics["accuracy"] == 0.5
        # ...but average precision exposes it, which is why it is the primary metric.
        assert metrics["average_precision"] < 0.5

    def test_perfect_ranking_scores_full_average_precision(self) -> None:
        """AP rewards correct ordering even when the 0.5 threshold is wrong."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.zeros(4, dtype=int)  # every prediction below threshold
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])  # but perfectly ordered

        metrics = classification_metrics(y_true, y_pred, y_proba)

        assert metrics["recall"] == 0.0  # thresholded output is still useless
        assert metrics["average_precision"] == 1.0  # ranking is flawless
        assert metrics["roc_auc"] == 1.0

    def test_regression_metrics_on_perfect_prediction(self) -> None:
        y_true = np.array([5.0, 6.0, 7.0])
        metrics = regression_metrics(y_true, y_true)
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0

    def test_regression_metrics_known_error(self) -> None:
        y_true = np.array([5.0, 5.0])
        y_pred = np.array([6.0, 4.0])
        metrics = regression_metrics(y_true, y_pred)
        assert metrics["mae"] == 1.0
        assert metrics["rmse"] == 1.0
        assert metrics["max_error"] == 1.0
        assert metrics["mean_residual"] == 0.0

    def test_metrics_to_display_formats_floats(self) -> None:
        formatted = metrics_to_display({"roc_auc": 0.88123456, "count": 5, "cm": [[1, 2]]})
        assert formatted["roc_auc"] == "0.8812"
        assert formatted["count"] == "5"
        assert formatted["cm"] == "matrix"


class TestArtifactPersistence:
    """Artifacts must round-trip and fail loudly when absent."""

    def test_save_and_load_round_trip(self, fresh_settings, raw_frame: pd.DataFrame) -> None:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline

        from wine_quality.features.pipeline import build_preprocessor

        features = raw_frame[list(FEATURE_NAMES)]
        target = (raw_frame["quality"] >= 7).astype(int)

        classifier = Pipeline(
            [("preprocessor", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))]
        ).fit(features, target)
        regressor = Pipeline(
            [("preprocessor", build_preprocessor()), ("model", Ridge())]
        ).fit(features, raw_frame["quality"])

        metadata = ModelMetadata(
            model_version="test",
            trained_at=ModelMetadata.now_iso(),
            good_quality_cutoff=7,
            feature_names=list(FEATURE_NAMES),
            n_training_rows=len(features),
            n_test_rows=0,
            classifier_algorithm="LogisticRegression",
            regressor_algorithm="Ridge",
        )

        paths = save_artifacts(classifier, regressor, metadata, fresh_settings)
        assert all(path for path in paths.values())
        assert artifacts_exist(fresh_settings)

        loaded_classifier, loaded_regressor, loaded_metadata = load_artifacts(fresh_settings)
        assert loaded_metadata.model_version == "test"
        assert loaded_metadata.feature_names == list(FEATURE_NAMES)

        # The reloaded pipelines must produce identical predictions.
        sample = features.head(5)
        np.testing.assert_array_equal(
            loaded_classifier.predict(sample), classifier.predict(sample)
        )
        np.testing.assert_allclose(loaded_regressor.predict(sample), regressor.predict(sample))

    def test_load_without_artifacts_raises(self, fresh_settings) -> None:
        with pytest.raises(ArtifactError, match="Missing model artifacts"):
            load_artifacts(fresh_settings)

    def test_artifacts_exist_is_false_when_empty(self, fresh_settings) -> None:
        assert artifacts_exist(fresh_settings) is False

    def test_metadata_serialises_to_json(self) -> None:
        metadata = ModelMetadata(
            model_version="0.1.0",
            trained_at=ModelMetadata.now_iso(),
            good_quality_cutoff=7,
            feature_names=list(FEATURE_NAMES),
            n_training_rows=1279,
            n_test_rows=320,
            classifier_algorithm="RandomForestClassifier",
            regressor_algorithm="RandomForestRegressor",
        )
        payload = json.loads(json.dumps(metadata.to_dict()))
        assert payload["good_quality_cutoff"] == 7
        assert payload["feature_names"] == list(FEATURE_NAMES)


class TestRunLog:
    """The JSONL experiment log must append and read back cleanly."""

    def test_append_and_read(self, fresh_settings) -> None:
        append_run_log({"run": 1, "roc_auc": 0.88}, fresh_settings)
        append_run_log({"run": 2, "roc_auc": 0.90}, fresh_settings)

        records = read_run_log(fresh_settings)
        assert len(records) == 2
        assert records[0]["run"] == 1
        assert records[1]["roc_auc"] == 0.90

    def test_read_empty_log_returns_empty_list(self, fresh_settings) -> None:
        assert read_run_log(fresh_settings) == []


@pytest.mark.slow
class TestTrainAll:
    """End-to-end training. Slow: cross-validates six pipelines."""

    @pytest.fixture(scope="class")
    def result(self, isolated_settings):
        return train_all(settings=isolated_settings, persist=True)

    def test_produces_both_pipelines(self, result) -> None:
        assert result.classifier is not None
        assert result.regressor is not None

    def test_classifier_beats_random_baseline(self, result) -> None:
        """The Kaggle card suggests ~0.88 AUC is achievable without tuning."""
        assert result.roc_auc > 0.80, f"ROC-AUC {result.roc_auc:.3f} is below a useful threshold"

    def test_classifier_beats_no_skill_pr_baseline(self, result) -> None:
        """PR-AUC must clear the positive rate, or the model adds nothing."""
        assert result.average_precision > 0.40

    def test_regressor_beats_predicting_the_mean(self, result) -> None:
        """R² > 0 means the regressor explains variance beyond the mean."""
        assert result.regression_metrics["r2"] > 0.20

    def test_rmse_is_within_one_score_point(self, result) -> None:
        assert result.rmse < 1.0

    def test_metadata_records_provenance(self, result) -> None:
        assert result.metadata.good_quality_cutoff == 7
        assert result.metadata.feature_names == list(FEATURE_NAMES)
        assert result.metadata.n_training_rows + result.metadata.n_test_rows == 1599
        assert result.metadata.classifier_algorithm
        assert result.metadata.regressor_algorithm

    def test_artifacts_written_to_disk(self, result, isolated_settings) -> None:
        assert artifacts_exist(isolated_settings)
        assert isolated_settings.metrics_path.exists()
        assert isolated_settings.runs_log_path.exists()

    def test_run_log_has_one_record(self, result, isolated_settings) -> None:
        records = read_run_log(isolated_settings)
        assert len(records) == 1
        assert records[0]["cutoff"] == 7

    def test_feature_importance_covers_all_features(self, result) -> None:
        assert result.feature_importance is not None
        assert set(result.feature_importance.keys()) == set(FEATURE_NAMES)

    def test_feature_importance_is_sorted_descending(self, result) -> None:
        values = list(result.feature_importance.values())
        assert values == sorted(values, reverse=True)

    def test_alcohol_is_a_top_predictor(self, result) -> None:
        """Domain knowledge check: alcohol is the strongest known signal."""
        top_three = list(result.feature_importance.keys())[:3]
        assert "alcohol" in top_three, f"expected alcohol in top 3, got {top_three}"

    def test_predictions_are_in_valid_range(self, result, raw_frame: pd.DataFrame) -> None:
        predictions = result.regressor.predict(raw_frame[list(FEATURE_NAMES)])
        assert predictions.min() > 2.0
        assert predictions.max() < 9.0

    def test_probabilities_are_valid(self, result, raw_frame: pd.DataFrame) -> None:
        probabilities = result.classifier.predict_proba(raw_frame[list(FEATURE_NAMES)])[:, 1]
        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0

    def test_no_persist_leaves_no_artifacts(self, fresh_settings) -> None:
        train_all(settings=fresh_settings, persist=False)
        assert not artifacts_exist(fresh_settings)


class TestFeatureImportance:
    """Importance extraction must handle both trees and linear models."""

    def test_returns_none_for_estimator_without_importance(self) -> None:
        from sklearn.pipeline import Pipeline

        from wine_quality.features.pipeline import build_preprocessor

        class Opaque:
            def fit(self, x, y):  # noqa: ANN001, ANN201
                return self

            def predict(self, x):  # noqa: ANN001, ANN201
                return np.zeros(len(x))

        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", Opaque())])
        assert feature_importance(pipeline) is None