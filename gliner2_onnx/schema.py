"""Schema builder for multi-task GLiNER2 extraction."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class EntityConfig:
    entity_types: list[str]
    threshold: float = 0.5


@dataclass(frozen=True)
class ClassificationConfig:
    task: str
    labels: list[str]
    threshold: float = 0.5
    multi_label: bool = False


@dataclass(frozen=True)
class Schema:
    """
    Immutable builder for multi-task GLiNER2 extraction schemas.

    Combine NER and multiple classification tasks into a single encoder forward pass.

    Example:
        >>> schema = (
        ...     Schema()
        ...     .entities(["person", "email"], threshold=0.5)
        ...     .classification("safety", ["safe", "unsafe"])
        ...     .classification("intent", ["informational", "adversarial"], multi_label=True)
        ... )
        >>> result = runtime.extract(text, schema)
        >>> result.entities        # list[Entity]
        >>> result.classifications  # {"safety": {"safe": 0.92}, "intent": {...}}
    """

    _entity_config: EntityConfig | None = field(default=None, repr=False)
    _classification_configs: tuple[ClassificationConfig, ...] = field(
        default_factory=tuple, repr=False
    )

    def entities(self, entity_types: list[str], threshold: float = 0.5) -> Schema:
        """Configure NER extraction. Only one entities() call per schema is supported."""
        if not entity_types:
            raise ValueError("entity_types cannot be empty")
        return replace(
            self,
            _entity_config=EntityConfig(list(entity_types), threshold),
        )

    def classification(
        self,
        task: str,
        labels: list[str],
        threshold: float = 0.5,
        multi_label: bool = False,
    ) -> Schema:
        """Add a classification task. Multiple classification() calls are supported."""
        if not task or not task.strip():
            raise ValueError("task name cannot be empty")
        if not labels:
            raise ValueError("labels cannot be empty")
        if task in {c.task for c in self._classification_configs}:
            raise ValueError(f"Duplicate classification task name: '{task}'")
        return replace(
            self,
            _classification_configs=self._classification_configs
            + (ClassificationConfig(task, list(labels), threshold, multi_label),),
        )

    @property
    def has_entities(self) -> bool:
        return self._entity_config is not None

    @property
    def has_classifications(self) -> bool:
        return bool(self._classification_configs)

    def validate(self) -> None:
        """Raise ValueError if no tasks are configured."""
        if not self.has_entities and not self.has_classifications:
            raise ValueError(
                "Schema has no tasks. Call .entities() and/or .classification() first."
            )
