#!/usr/bin/env python3
"""Unit tests for Schema builder — no model required."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gliner2_onnx.schema import ClassificationConfig, EntityConfig, Schema


class TestSchemaBuilder:
    """Test Schema immutable builder pattern."""

    def test_empty_schema(self) -> None:
        s = Schema()
        assert not s.has_entities
        assert not s.has_classifications

    def test_entities(self) -> None:
        s = Schema().entities(["person", "org"])
        assert s.has_entities
        assert s._entity_config is not None
        assert s._entity_config.entity_types == ["person", "org"]
        assert s._entity_config.threshold == 0.5

    def test_entities_custom_threshold(self) -> None:
        s = Schema().entities(["person"], threshold=0.7)
        assert s._entity_config.threshold == 0.7

    def test_classification(self) -> None:
        s = Schema().classification("safety", ["safe", "unsafe"])
        assert s.has_classifications
        assert len(s._classification_configs) == 1
        cc = s._classification_configs[0]
        assert cc.task == "safety"
        assert cc.labels == ["safe", "unsafe"]
        assert cc.threshold == 0.5
        assert cc.multi_label is False

    def test_classification_multi_label(self) -> None:
        s = Schema().classification("topics", ["news", "sport"], multi_label=True)
        assert s._classification_configs[0].multi_label is True

    def test_chaining(self) -> None:
        s = (
            Schema()
            .entities(["person", "email"])
            .classification("safety", ["safe", "unsafe"])
            .classification("intent", ["info", "adversarial"], multi_label=True)
        )
        assert s.has_entities
        assert s.has_classifications
        assert len(s._classification_configs) == 2
        assert s._classification_configs[0].task == "safety"
        assert s._classification_configs[1].task == "intent"

    def test_immutability(self) -> None:
        base = Schema()
        with_entities = base.entities(["person"])
        with_class = base.classification("safety", ["safe", "unsafe"])

        # base unchanged
        assert not base.has_entities
        assert not base.has_classifications
        # with_entities has no classification
        assert not with_entities.has_classifications
        # with_class has no entities
        assert not with_class.has_entities

    def test_entities_overwrite(self) -> None:
        s = Schema().entities(["person"]).entities(["org", "location"])
        assert s._entity_config.entity_types == ["org", "location"]

    def test_classification_order_preserved(self) -> None:
        tasks = ["safety", "intent", "tone", "harmful"]
        s = Schema()
        for task in tasks:
            s = s.classification(task, ["a", "b"])
        assert [cc.task for cc in s._classification_configs] == tasks

    def test_input_list_is_copied(self) -> None:
        labels = ["safe", "unsafe"]
        s = Schema().classification("safety", labels)
        labels.append("unknown")
        assert s._classification_configs[0].labels == ["safe", "unsafe"]

    def test_entity_types_list_is_copied(self) -> None:
        entity_types = ["person"]
        s = Schema().entities(entity_types)
        entity_types.append("org")
        assert s._entity_config.entity_types == ["person"]


class TestSchemaValidation:
    """Test Schema validation errors."""

    def test_validate_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no tasks"):
            Schema().validate()

    def test_validate_with_entities_ok(self) -> None:
        Schema().entities(["person"]).validate()

    def test_validate_with_classification_ok(self) -> None:
        Schema().classification("safety", ["safe", "unsafe"]).validate()

    def test_validate_with_both_ok(self) -> None:
        Schema().entities(["person"]).classification("safety", ["safe", "unsafe"]).validate()

    def test_empty_entity_types_raises(self) -> None:
        with pytest.raises(ValueError, match="entity_types cannot be empty"):
            Schema().entities([])

    def test_empty_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="labels cannot be empty"):
            Schema().classification("safety", [])

    def test_empty_task_name_raises(self) -> None:
        with pytest.raises(ValueError, match="task name cannot be empty"):
            Schema().classification("", ["safe", "unsafe"])

    def test_blank_task_name_raises(self) -> None:
        with pytest.raises(ValueError, match="task name cannot be empty"):
            Schema().classification("   ", ["safe", "unsafe"])

    def test_duplicate_task_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate classification task name"):
            Schema().classification("safety", ["a", "b"]).classification("safety", ["c", "d"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
