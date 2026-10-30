"""Run local classifiers with the frozen task input contract."""

import json
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..datasets import DatasetSpec
from .encoding import TOKENS, encode_rows

BASELINE_OPTIONS = {
    "xgboost": frozenset(),
    "tabpfn": frozenset({"checkpoint_path", "allow_download"}),
    "tabfm": frozenset(
        {"checkpoint_path", "allow_download", "n_estimators", "batch_size"}
    ),
}


def check_options(model: str, options: dict[str, Any]) -> None:
    """Reject options that the named local baseline does not accept."""
    if model not in BASELINE_OPTIONS:
        raise ValueError(f"Unknown baseline: {model}")
    unknown = set(options) - BASELINE_OPTIONS[model]
    if unknown:
        raise ValueError(f"Unknown {model} baseline options: {sorted(unknown)}")
    if model == "xgboost" and options:
        raise ValueError("XGBoost uses fixed settings; options must be empty")


def _classifier(
    model: str, spec: DatasetSpec, n_rows: int, n_features: int,
    seed: int, device: str, options: dict[str, Any],
) -> Any:
    """Construct one classifier without changing its training data."""
    check_options(model, options)
    if model == "xgboost":
        cls = import_module("xgboost").XGBClassifier
        params = {
            "objective": "binary:logistic" if len(spec.labels) == 2 else "multi:softprob",
            "eval_metric": "logloss" if len(spec.labels) == 2 else "mlogloss",
            "random_state": seed,
            "device": device,
            # Row and column subsampling make each replicate seed-dependent.
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        if len(spec.labels) > 2:
            params["num_class"] = len(spec.labels)
        return cls(**params)
    if model == "tabpfn":
        checkpoint = options.get("checkpoint_path")
        if checkpoint is None and not options.get("allow_download", False):
            raise ValueError("Set TabPFN checkpoint_path or explicitly allow_download")
        kwargs = {"device": device, "random_state": seed,
                  "categorical_features_indices": list(range(spec.max_length))}
        if checkpoint is not None:
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(checkpoint)
            kwargs["model_path"] = str(checkpoint)
        cls = import_module("tabpfn").TabPFNClassifier
        version = import_module("tabpfn.constants").ModelVersion.V2
        return cls.create_default_for_version(version, **kwargs)
    if model == "tabfm":
        checkpoint = options.get("checkpoint_path")
        if checkpoint is None and not options.get("allow_download", False):
            raise ValueError("Set TabFM checkpoint_path or explicitly allow_download")
        if checkpoint is not None and not Path(checkpoint).is_dir():
            raise FileNotFoundError(checkpoint)
        package = import_module("tabfm")
        torch = import_module("torch")
        torch.manual_seed(seed)
        pretrained = package.tabfm_v1_0_0_pytorch.load(
            model_type="classification", checkpoint_path=checkpoint, device=device,
            use_cache=False,
        )
        return package.TabFMClassifier(
            model=pretrained, max_num_rows=n_rows, max_num_features=n_features,
            random_state=seed, **{key: value for key, value in options.items()
                                  if key in {"n_estimators", "batch_size"}},
        )


def predict_baseline(
    model: str, spec: DatasetSpec, train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]], seed: int, device: str = "cpu",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a local classifier and predict ordered labels.

    Args:
        model: xgboost, tabpfn, or tabfm.
        spec: Task inputs and allowed labels.
        train_rows: Complete selected training partition.
        test_rows: Ordered test inputs.
        seed: Recorded model seed.
        device: CPU or CUDA device.
        options: Checkpoint or explicit download settings for foundation models.

    Returns:
        Predictions and model settings with execution timing.

    Raises:
        ValueError: If labels, outputs, or full-context checks fail.
    """
    if not train_rows or not test_rows:
        raise ValueError("Training and test rows must be nonempty")
    started = perf_counter()
    x_train = encode_rows(spec, train_rows, one_hot=model == "xgboost")
    x_test = encode_rows(spec, test_rows, one_hot=model == "xgboost")
    label_ids = {label: index for index, label in enumerate(spec.labels)}
    try:
        y_train = np.array([label_ids[row[spec.target]] for row in train_rows])
    except KeyError as error:
        raise ValueError(f"Missing or invalid training label: {error}") from error
    if set(y_train.tolist()) != set(range(len(spec.labels))):
        raise ValueError("Training data must contain every declared class")
    try:
        classifier = _classifier(
            model, spec, len(train_rows), x_train.shape[1], seed, device, options or {},
        )
    except ImportError as error:
        raise RuntimeError(f"Install the optional {model} dependencies") from error
    classifier.fit(x_train, y_train)
    if model == "tabpfn" and classifier.n_train_samples_ != len(train_rows):
        raise ValueError("TabPFN did not retain the complete training context")
    if model == "tabfm":
        generator = classifier.ensemble_generator_
        patterns = generator.row_subsample_patterns_
        if len(generator.y_) != len(train_rows) or any(
            pattern is not None for group in patterns.values() for pattern in group
        ):
            raise ValueError("TabFM did not retain the complete training context")
    raw_predictions = np.asarray(classifier.predict(x_test)).reshape(-1)
    if len(raw_predictions) != len(test_rows):
        raise ValueError("Baseline returned an incorrect prediction count")
    predictions = []
    for value in raw_predictions:
        index = int(value)
        if value != index or index not in range(len(spec.labels)):
            raise ValueError(f"Invalid predicted label: {value}")
        predictions.append(spec.labels[index])
    return {
        "predictions": predictions,
        "metadata": {
            "model": model,
            "checkpoint_family": "v2" if model == "tabpfn" else None,
            "seed": seed, "device": device, "training_rows": len(train_rows),
            "test_rows": len(test_rows), "feature_count": x_train.shape[1],
            "feature_inputs": list(spec.inputs), "token_vocabulary": list(TOKENS),
            "encoding": "positional_one_hot" if model == "xgboost" else "positional_integer",
            "options": options or {}, "elapsed_seconds": perf_counter() - started,
            "effective_parameters": json.loads(classifier.get_booster().save_config())["learner"]
            if model == "xgboost" else None,
        },
    }
