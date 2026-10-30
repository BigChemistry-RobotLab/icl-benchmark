"""Check sequence encoding, dataset contracts, and local classifier interfaces."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.datasets import DatasetSpec, dataset_spec
from experiments.models import local as baselines
from experiments.models.encoding import (
    PAD,
    TOKEN_IDS,
    TOKENS,
    UNK,
    encode_rows,
    tokenize,
)
from experiments.models.profile import profile_model
from icl_agent.config import Task


@pytest.fixture()
def spec() -> DatasetSpec:
    """Build the surface-tension contract without reading repository data."""
    return DatasetSpec(
        name="surface_tension",
        inputs=("sequence", "pH", "tail_length"),
        target="surface_active",
        labels=("false", "true"),
        max_length=3,
    )


def _rows() -> list[dict[str, str]]:
    """Return labeled examples with an audit-only column."""
    return [
        {"sequence": "AAA", "pH": "7", "tail_length": "8",
         "surface_tension": "999", "surface_active": "false"},
        {"sequence": "GGG", "pH": "7", "tail_length": "8",
         "surface_tension": "999", "surface_active": "true"},
    ] * 4


def test_dataset_spec_keeps_only_non_context_inputs() -> None:
    """Derive the model input contract from a validated task."""
    task = Task.model_validate({
        "inputs": [{"column": "sequence"}, {"column": "notes", "type": "context"}],
        "outputs": [{"column": "label"}],
        "additional_context": "Predict the label.",
        "allowed_labels": {"label": ["0", "1"]},
    })
    assert dataset_spec(task, "toy", 5) == DatasetSpec(
        name="toy",
        inputs=("sequence",),
        target="label",
        labels=("0", "1"),
        max_length=5,
    )


@pytest.mark.parametrize(("name", "max_length", "inputs", "outputs", "labels", "message"), [
    (" ", 5, [{"column": "sequence"}], [{"column": "label"}], {"label": ["0"]},
     "name must not be blank"),
    ("toy", 0, [{"column": "sequence"}], [{"column": "label"}], {"label": ["0"]},
     "length must be positive"),
    ("toy", 5, [{"column": "sequence"}],
     [{"column": "label"}, {"column": "other"}],
     {"label": ["0"], "other": ["a"]}, "exactly one output"),
    ("toy", 5, [{"column": "notes", "type": "context"}],
     [{"column": "label"}], {"label": ["0"]}, "sequence input"),
    ("toy", 5, [{"column": "sequence"}], [{"column": "label"}], {}, "allowed labels"),
])
def test_dataset_spec_rejects_unsupported_tasks(
    name: str, max_length: int, inputs: list[dict[str, str]],
    outputs: list[dict[str, str]], labels: dict[str, list[str]], message: str,
) -> None:
    """Reject tasks without a usable single-label sequence contract."""
    task = Task.model_validate({
        "inputs": inputs,
        "outputs": outputs,
        "allowed_labels": labels,
    })
    with pytest.raises(ValueError, match=message):
        dataset_spec(task, name, max_length)


def test_tokens_and_padding(spec: DatasetSpec) -> None:
    """Encode only the declared inputs and keep unknown columns out."""
    assert tokenize("aP'G") == ["a", "P'", "G"]
    rows = [{"sequence": "aP'", "pH": "7", "tail_length": "8",
             "surface_tension": "999", "surface_active": "true"}]
    integer = encode_rows(spec, rows)
    assert integer.tolist() == [[TOKEN_IDS["a"], TOKEN_IDS["P'"], PAD, 7, 8]]
    onehot = encode_rows(spec, rows, one_hot=True)
    assert onehot.shape == (1, 3 * len(TOKENS) + 2)
    assert onehot[0, :3 * len(TOKENS)].sum() == 3
    assert encode_rows(spec, [{**rows[0], "surface_tension": "-1"}]).tolist() == (
        integer.tolist()
    )
    assert PAD != UNK != TOKEN_IDS["A"] != TOKEN_IDS["a"]


def test_invalid_sequence_and_numeric_inputs(spec: DatasetSpec) -> None:
    """Reject oversized sequences and non-finite numeric inputs."""
    with pytest.raises(ValueError, match="length"):
        encode_rows(spec, [{"sequence": "AAAA", "pH": "7", "tail_length": "8"}])
    with pytest.raises(ValueError, match="Non-finite"):
        encode_rows(spec, [{"sequence": "AAA", "pH": "nan", "tail_length": "8"}])


def test_xgboost_uses_seeded_subsampling(
    spec: DatasetSpec, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give XGBoost stochastic settings that consume the replicate seed."""
    calls = {}

    class FakeClassifier:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(
        baselines, "import_module",
        lambda name: SimpleNamespace(XGBClassifier=FakeClassifier),
    )
    baselines._classifier("xgboost", spec, 8, 5, 7, "cpu", {})
    assert calls["random_state"] == 7
    assert calls["subsample"] == 0.8
    assert calls["colsample_bytree"] == 0.8


def test_real_xgboost_profile(spec: DatasetSpec) -> None:
    """Fit one real local model and record hardware measurements."""
    pytest.importorskip("xgboost")
    result = profile_model("xgboost", spec, _rows(), _rows()[:2])
    assert len(result["predictions"]) == 2
    assert set(result["predictions"]) <= {"false", "true"}
    assert result["metadata"]["encoding"] == "positional_one_hot"
    assert result["profile"]["process_peak_rss_bytes"] > 0


def test_foundation_models_do_not_download_by_default(spec: DatasetSpec) -> None:
    """Require an explicit checkpoint or download permission."""
    for model in ("tabpfn", "tabfm"):
        with pytest.raises(ValueError, match="allow_download"):
            baselines.predict_baseline(model, spec, _rows(), _rows()[:2], 42)


def test_tabpfn_selects_v2(spec: DatasetSpec, monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the current TabPFN version for the complete training context."""
    calls = {}

    class FakeClassifier:
        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            calls.update(version=version, **kwargs)
            return cls()

        def fit(self, x, y):
            calls["rows"] = len(x)
            self.n_train_samples_ = len(x)

        def predict(self, x):
            return np.zeros(len(x), dtype=int)

    packages = {
        "tabpfn": SimpleNamespace(TabPFNClassifier=FakeClassifier),
        "tabpfn.constants": SimpleNamespace(ModelVersion=SimpleNamespace(V2="V2")),
    }
    monkeypatch.setattr(baselines, "import_module", packages.__getitem__)
    baselines.predict_baseline("tabpfn", spec, _rows(), _rows()[:2], 7,
                               options={"allow_download": True})
    assert calls["version"] == "V2"
    assert calls["categorical_features_indices"] == [0, 1, 2]
    assert calls["rows"] == 8


@pytest.mark.parametrize("sampled", [False, True])
def test_tabfm_full_context(spec: DatasetSpec, monkeypatch: pytest.MonkeyPatch,
                            sampled: bool) -> None:
    """Reject a TabFM fit that subsamples the supplied training rows."""
    calls = {}

    class FakeClassifier:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def fit(self, x, y):
            self.ensemble_generator_ = SimpleNamespace(
                y_=y,
                row_subsample_patterns_={
                    "none": [np.array([0]) if sampled else None],
                },
            )

        def predict(self, x):
            return np.zeros(len(x), dtype=int)

    def load(**kwargs):
        calls["load_kwargs"] = kwargs
        return object()

    packages = {
        "tabfm": SimpleNamespace(
            TabFMClassifier=FakeClassifier,
            tabfm_v1_0_0_pytorch=SimpleNamespace(load=load),
        ),
        "torch": SimpleNamespace(manual_seed=lambda seed: None),
    }
    monkeypatch.setattr(baselines, "import_module", packages.__getitem__)
    if sampled:
        with pytest.raises(ValueError, match="complete training"):
            baselines.predict_baseline("tabfm", spec, _rows(), _rows()[:2], 42,
                                       options={"allow_download": True})
    else:
        result = baselines.predict_baseline("tabfm", spec, _rows(), _rows()[:2], 42,
                                            options={"allow_download": True})
        assert result["predictions"] == ["false", "false"]
        assert calls["max_num_rows"] == 8
        assert calls["max_num_features"] == 5
        assert calls["load_kwargs"] == {
            "model_type": "classification", "checkpoint_path": None,
            "device": "cpu", "use_cache": False,
        }
