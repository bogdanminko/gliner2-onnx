#!/usr/bin/env python3
"""Test GLiNER2 ONNX runtime against pre-generated fixtures."""

import json
import sys
from pathlib import Path
from typing import TypedDict

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gliner2_onnx import GLiNER2ONNXRuntime, Schema

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_PATH = Path(__file__).parent / "gliner2.fixtures.json"
SCORE_TOLERANCE = 0.05


class ClassificationFixture(TypedDict):
    text: str
    labels: list[str]
    expected_label: str
    expected_score: float


class EntityFixture(TypedDict):
    text: str
    label: str


class NERFixture(TypedDict):
    text: str
    labels: list[str]
    threshold: float
    expected: list[EntityFixture]


class SchemaClassificationTask(TypedDict):
    task: str
    labels: list[str]
    multi_label: bool
    threshold: float


class SchemaFixture(TypedDict):
    text: str
    entity_labels: list[str]
    entity_threshold: float
    classification_tasks: list[SchemaClassificationTask]
    expected_entities: list[EntityFixture]
    expected_classifications: dict[str, dict[str, float]]


class ModelFixtures(TypedDict):
    classification: list[ClassificationFixture]
    ner: list[NERFixture]
    schema_extraction: list[SchemaFixture]


def load_fixtures() -> dict[str, ModelFixtures]:
    """Load fixtures from JSON file."""
    if not FIXTURES_PATH.exists():
        pytest.skip(f"Fixtures not found at {FIXTURES_PATH}. Run 'uv run python tests/generate_fixtures.py' first.")
    with FIXTURES_PATH.open() as f:
        return json.load(f)


def get_available_model_precisions() -> list[tuple[str, str]]:
    """Get list of (model_key, precision) tuples that are available for testing."""
    if not FIXTURES_PATH.exists():
        return []

    with FIXTURES_PATH.open() as f:
        fixtures: dict[str, ModelFixtures] = json.load(f)

    available = []
    for model_key in fixtures:
        model_path = PROJECT_ROOT / "model_out" / model_key
        config_path = model_path / "gliner2_config.json"

        if not config_path.exists():
            continue

        with config_path.open() as f:
            config = json.load(f)

        precisions = list(config.get("onnx_files", {}).keys())
        available.extend((model_key, p) for p in precisions)

    return available


available_model_precisions = get_available_model_precisions()
if not available_model_precisions:
    pytest.skip("No exported models found. Run export first.", allow_module_level=True)


@pytest.fixture(scope="module")
def fixtures() -> dict[str, ModelFixtures]:
    """Load all fixtures."""
    return load_fixtures()


@pytest.fixture(scope="module", params=available_model_precisions, ids=lambda x: f"{x[0]}-{x[1]}")
def model_setup(request: pytest.FixtureRequest, fixtures: dict[str, ModelFixtures]) -> tuple[str, str, GLiNER2ONNXRuntime, ModelFixtures]:
    """Setup runtime for each model and precision combination."""
    model_key, precision = request.param
    model_path = PROJECT_ROOT / "model_out" / model_key
    runtime = GLiNER2ONNXRuntime(str(model_path), precision=precision, providers=["CPUExecutionProvider"])
    return model_key, precision, runtime, fixtures[model_key]


class TestClassification:
    """Classification tests."""

    def test_classification(self, model_setup: tuple[str, str, GLiNER2ONNXRuntime, ModelFixtures]) -> None:
        """Test classification predictions match expected results."""
        model_key, precision, runtime, model_fixtures = model_setup
        if precision == "int8":
            pytest.skip("int8 fixtures require int8-specific ground truth; fp32 fixtures not comparable")

        for fixture in model_fixtures["classification"]:
            text = fixture["text"]
            labels = fixture["labels"]
            expected_label = fixture["expected_label"]
            expected_score = fixture["expected_score"]

            result = runtime.classify(text, labels)
            actual_label = next(iter(result.keys()))
            actual_score = result[actual_label]

            assert actual_label == expected_label, (
                f"[{model_key}/{precision}] Label mismatch for '{text[:50]}...'\nExpected: {expected_label}, Got: {actual_label}"
            )
            assert abs(actual_score - expected_score) <= SCORE_TOLERANCE, (
                f"[{model_key}/{precision}] Score mismatch for '{text[:50]}...'\nExpected: {expected_score:.4f}, Got: {actual_score:.4f}"
            )


class TestNER:
    """NER tests."""

    def test_ner(self, model_setup: tuple[str, str, GLiNER2ONNXRuntime, ModelFixtures]) -> None:
        """Test NER extraction matches expected entities."""
        model_key, precision, runtime, model_fixtures = model_setup
        if precision == "int8":
            pytest.skip("int8 fixtures require int8-specific ground truth; fp32 fixtures not comparable")

        for fixture in model_fixtures["ner"]:
            text = fixture["text"]
            labels = fixture["labels"]
            threshold = fixture["threshold"]
            expected = fixture["expected"]

            entities = runtime.extract_entities(text, labels, threshold=threshold)

            actual_set = {(e.text, e.label) for e in entities}
            expected_set = {(e["text"], e["label"]) for e in expected}

            assert actual_set == expected_set, (
                f"[{model_key}/{precision}] Entity mismatch for '{text[:50]}...'\nMissing: {expected_set - actual_set}\nExtra: {actual_set - expected_set}"
            )


class TestBatch:
    """Batch methods must return the same results as N individual calls."""

    def test_classify_batch_matches_single(
        self, model_setup: tuple[str, str, GLiNER2ONNXRuntime, ModelFixtures]
    ) -> None:
        _, precision, runtime, model_fixtures = model_setup
        if precision == "int8":
            pytest.skip("int8 batch consistency not guaranteed: padding changes dequant outputs")
        # Pick first 10 classification fixtures
        fixtures = model_fixtures["classification"][:10]
        if not fixtures:
            pytest.skip("No classification fixtures available")

        # Group by labels (batch requires same labels for all texts)
        from itertools import groupby
        key = lambda f: tuple(f["labels"])
        for labels_tuple, group in groupby(sorted(fixtures, key=key), key=key):
            group_list = list(group)
            texts = [f["text"] for f in group_list]
            labels = list(labels_tuple)

            single_results = [runtime.classify(t, labels) for t in texts]
            batch_results = runtime.classify_batch(texts, labels)

            assert len(batch_results) == len(single_results)
            for i, (single, batch) in enumerate(zip(single_results, batch_results)):
                assert single.keys() == batch.keys(), (
                    f"Label mismatch at index {i}: single={single}, batch={batch}"
                )
                for label in single:
                    assert abs(single[label] - batch[label]) < 2e-3, (
                        f"Score mismatch at index {i} label '{label}': "
                        f"single={single[label]:.6f}, batch={batch[label]:.6f}"
                    )

    def test_extract_entities_batch_matches_single(
        self, model_setup: tuple[str, str, GLiNER2ONNXRuntime, ModelFixtures]
    ) -> None:
        _, _, runtime, model_fixtures = model_setup
        fixtures = model_fixtures["ner"][:10]
        if not fixtures:
            pytest.skip("No NER fixtures available")

        from itertools import groupby
        key = lambda f: (tuple(f["labels"]), f["threshold"])
        for group_key, group in groupby(sorted(fixtures, key=key), key=key):
            group_list = list(group)
            labels, threshold = list(group_key[0]), group_key[1]
            texts = [f["text"] for f in group_list]

            single_results = [runtime.extract_entities(t, labels, threshold=threshold) for t in texts]
            batch_results = runtime.extract_entities_batch(texts, labels, threshold=threshold)

            assert len(batch_results) == len(single_results)
            for i, (single, batch) in enumerate(zip(single_results, batch_results)):
                single_set = {(e.text, e.label) for e in single}
                batch_set = {(e.text, e.label) for e in batch}
                assert single_set == batch_set, (
                    f"Entity mismatch at index {i}\nMissing: {single_set - batch_set}\nExtra: {batch_set - single_set}"
                )


class TestSchemaExtraction:
    """Schema multi-task extraction tests against PyTorch ground truth."""

    def test_schema_extraction(
        self, model_setup: tuple[str, str, GLiNER2ONNXRuntime, ModelFixtures]
    ) -> None:
        model_key, precision, runtime, model_fixtures = model_setup
        fixtures = model_fixtures.get("schema_extraction", [])
        if not fixtures:
            pytest.skip("No schema_extraction fixtures. Run generate_fixtures.py first.")

        for fixture in fixtures:
            schema = Schema()
            if fixture["entity_labels"]:
                schema = schema.entities(
                    entity_types=fixture["entity_labels"],
                    threshold=fixture["entity_threshold"],
                )
            for ct in fixture["classification_tasks"]:
                schema = schema.classification(
                    task=ct["task"],
                    labels=ct["labels"],
                    threshold=ct["threshold"],
                    multi_label=ct["multi_label"],
                )

            result = runtime.extract(fixture["text"], schema)

            # Check entities
            actual_entities = {(e.text, e.label) for e in result.entities}
            expected_entities = {(e["text"], e["label"]) for e in fixture["expected_entities"]}
            assert actual_entities == expected_entities, (
                f"[{model_key}/{precision}] Entity mismatch for '{fixture['text'][:50]}'\n"
                f"Missing: {expected_entities - actual_entities}\nExtra: {actual_entities - expected_entities}"
            )

            # Check classifications
            for ct in fixture["classification_tasks"]:
                task = ct["task"]
                assert task in result.classifications, (
                    f"[{model_key}/{precision}] Task '{task}' missing in classifications"
                )
                if not ct["multi_label"]:
                    actual_label = next(iter(result.classifications[task]))
                    expected_label = next(iter(fixture["expected_classifications"][task]))
                    assert actual_label == expected_label, (
                        f"[{model_key}/{precision}] Classification mismatch for task '{task}' "
                        f"on '{fixture['text'][:50]}'\nExpected: {expected_label}, Got: {actual_label}"
                    )

    def test_schema_extract_batch_matches_single(
        self, model_setup: tuple[str, str, GLiNER2ONNXRuntime, ModelFixtures]
    ) -> None:
        _, _, runtime, model_fixtures = model_setup
        fixtures = model_fixtures.get("schema_extraction", [])
        if not fixtures:
            pytest.skip("No schema_extraction fixtures. Run generate_fixtures.py first.")

        # Build one schema from first fixture (all fixtures share the same schema structure)
        first = fixtures[0]
        schema = Schema()
        if first["entity_labels"]:
            schema = schema.entities(first["entity_labels"], threshold=first["entity_threshold"])
        for ct in first["classification_tasks"]:
            schema = schema.classification(
                task=ct["task"], labels=ct["labels"],
                threshold=ct["threshold"], multi_label=ct["multi_label"],
            )

        texts = [f["text"] for f in fixtures]
        single_results = [runtime.extract(t, schema) for t in texts]
        batch_results = runtime.extract_batch(texts, schema)

        assert len(batch_results) == len(single_results)
        for i, (single, batch) in enumerate(zip(single_results, batch_results)):
            single_entities = {(e.text, e.label) for e in single.entities}
            batch_entities = {(e.text, e.label) for e in batch.entities}
            assert single_entities == batch_entities, f"Entity mismatch at index {i}"
            assert single.classifications.keys() == batch.classifications.keys(), (
                f"Classification task mismatch at index {i}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
