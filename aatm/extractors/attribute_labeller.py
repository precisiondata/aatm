"""Label entity attributes with a decision model, with probabilities.

An attribute with a closed list of labels, such as ``smoking_status.status`` or the
ECOG grade, is a closed-set question. `DecisionAttributeLabeller` asks it of a
decision model (`aatm.llm.DecisionModel`) for each mention the extractor found, and
gets back the label with a probability.

Opt in by giving the labeller to the extractor; without it, the extractor's own LLM
call chooses the labels, as before:

```python
from aatm.extractors import DecisionAttributeLabeller, SimpleLLMExtractor

extractor = SimpleLLMExtractor(
    model_id="openai/gpt-5-mini",
    entity_definitions=definitions,
    attribute_labeller=DecisionAttributeLabeller(
        "openrouter/typesafe/jev-1.13",
        mode="verify",
        min_probability=0.8,
    ),
)
```

The labeller decides *who* labels; the entity definitions say *what kind of
question* each attribute is. An attribute marked ``ordered: true`` (its labels are a
scale, lowest first, like the ECOG grades) is asked as a score question; any other
as a choice question. Every question can also be answered "not stated", since a
mention need not state every attribute.

Modes:

- ``verify`` (default): the extractor labels as before, and the decision model
  checks each label. A label is kept only when both agree.
- ``replace``: the extractor's prompt leaves out the labelled attributes, so it only
  finds the mentions, and the decision model's labels are used.

Labels the decision model is not sure of (below ``min_probability``), or that the
two models disagree on, are not written to the entity: they go to the result's
``attribute_reviews`` for a person to check. When such an attribute is required, the
mention is rejected (``required_attribute_under_review``), with its children.

**Data governance:** each mention and its line of the note are sent to the decision
provider.
"""

import asyncio
import logging
import re
from typing import Any, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from aatm.data_models.extractors import (
    AttributeDefinition,
    AttributeIssue,
    AttributeReview,
    AttributeReviewReason,
    EntityDefinition,
    EntityDefinitions,
    EntityValidationResult,
    ExtractedAttribute,
    ExtractedEntity,
    ExtractedEntityCandidates,
    RejectedEntityCandidate,
    RejectionReason,
    TextSegment,
)

logger = logging.getLogger(__name__)

LabellerMode = Literal["verify", "replace"]

#: The answer for an attribute the note does not state for the mention.
NOT_STATED = "not_stated"

DEFAULT_INSTRUCTIONS = (
    "The state is one mention of {entity} in a clinical note: `mention` is its "
    "text, and `line` the line of the note it appears on. Answer about this mention "
    "only, from what the note states, without guessing. {attribute}"
)


class AttributeDecision(BaseModel):
    """The decision model's answer for one attribute of one mention.

    Attributes:
        label: The label, or ``None`` when the attribute is not stated.
        probability: The probability that this answer is right, when the model
            gives probabilities.
        probabilities: The probability of each label, and of `NOT_STATED`.
        refused: Whether the model declined to answer.
    """

    label: Optional[str] = None
    probability: Optional[float] = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    refused: bool = False


def _not_stated_option(attribute: AttributeDefinition) -> str:
    """`NOT_STATED`, or a variant when a label already uses that value."""
    values = {label.value.casefold() for label in attribute.labels}
    token = NOT_STATED
    while token.casefold() in values:
        token += "_"
    return token


def _uses_score(attribute: AttributeDefinition) -> bool:
    from aatm.llm.decisions import MAX_LEVELS

    return attribute.ordered and 2 <= len(attribute.labels) <= MAX_LEVELS


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,55}$")


class _AttributeQuestions:
    """The questions asked for one attribute, and how to read their answers."""

    def __init__(
        self,
        index: int,
        entity: EntityDefinition,
        attribute: AttributeDefinition,
        instructions: str,
    ):
        from aatm.llm.decisions import Option, Question

        self.attribute = attribute
        name = attribute.key if _NAME.match(attribute.key) else f"a{index}"
        self.not_stated = _not_stated_option(attribute)
        base = instructions.format(entity=entity.name, attribute=attribute.description)
        options = [
            Option(value=label.value, description=label.description)
            for label in attribute.labels
        ]
        self.questions: list[Question]
        if _uses_score(attribute):
            self.stated = Question.predicate(
                f"{name}_stated",
                f"{base} Does the note state this for the mention?",
            )
            self.score = Question.score(
                name,
                f"{base} If the note states it for the mention, which level does "
                "it give? The levels go from lowest to highest.",
                options,
            )
            self.questions = [self.stated, self.score]
        else:
            options.append(
                Option(
                    value=self.not_stated,
                    description="The note does not state this for this mention.",
                )
            )
            self.choice = Question.choice(
                name,
                f"{base} Choose the label the note states for this mention, or "
                f"`{self.not_stated}` when it does not state one.",
                options,
            )
            self.questions = [self.choice]

    def read(self, decision: Any) -> AttributeDecision:
        if _uses_score(self.attribute):
            return self._read_score(decision)
        answer = decision[self.choice.name]
        if answer.refused or answer.value is None:
            return AttributeDecision(refused=True)
        probabilities = {
            (NOT_STATED if k == self.not_stated else k): v
            for k, v in answer.probabilities.items()
        }
        label = None if answer.value == self.not_stated else str(answer.value)
        return AttributeDecision(
            label=label,
            probability=answer.probability,
            probabilities=probabilities,
        )

    def _read_score(self, decision: Any) -> AttributeDecision:
        stated = decision[self.stated.name]
        score = decision[self.score.name]
        if stated.refused or stated.value is None:
            return AttributeDecision(refused=True)
        p_stated = stated.probability
        if not stated.value:
            return AttributeDecision(
                label=None,
                probability=None if p_stated is None else 1.0 - p_stated,
                probabilities={}
                if p_stated is None
                else {
                    **{k: p_stated * v for k, v in score.probabilities.items()},
                    NOT_STATED: 1.0 - p_stated,
                },
            )
        if score.refused or score.label is None:
            return AttributeDecision(refused=True)
        p_level = score.probability
        probabilities: dict[str, float] = {}
        if p_stated is not None and score.probabilities:
            probabilities = {k: p_stated * v for k, v in score.probabilities.items()}
            probabilities[NOT_STATED] = 1.0 - p_stated
        return AttributeDecision(
            label=score.label,
            probability=None
            if p_stated is None or p_level is None
            else p_stated * p_level,
            probabilities=probabilities,
        )


class DecisionAttributeLabeller:
    """Label entity attributes with a decision model (`aatm.llm.DecisionModel`).

    Give it to `aatm.extractors.SimpleLLMExtractor` as ``attribute_labeller``, or to
    `aatm.omop.OmopExtractionPipeline`, which passes it on. See the module docs.

    For each mention, the model reads the entity's name, the mention's text and its
    line of the note, and answers one question per labelled attribute of the entity,
    in one call.

    Args:
        model: A decision model id for `aatm.llm.create_decision_model`, e.g.
            ``"openrouter/typesafe/jev-1.13"``, or a `DecisionModel`. Pin a version.
        mode: ``"verify"`` (default) or ``"replace"``, see the module docs.
        attributes: Which attributes to label: entity keys (all of the entity's
            attributes) or ``"<entity key>.<attribute key>"``. ``None`` (default):
            every attribute of every entity.
        min_probability: Labels the model gives a lower probability go to review
            instead of the entity. ``None`` (default): keep every label, recording
            its probability. Models without probabilities (``llm/``) then send
            every label to review.
        instructions: The text that starts every question. ``{entity}`` is replaced
            by the entity's name and ``{attribute}`` by the attribute's description.
        max_concurrency: Mentions labelled at the same time.
        model_kwargs: Passed to `create_decision_model` (``api_key``, ``base_url``,
            ``settings``) when `model` is an id.
    """

    def __init__(
        self,
        model: Any,
        mode: LabellerMode = "verify",
        attributes: Optional[Sequence[str]] = None,
        min_probability: Optional[float] = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        max_concurrency: int = 8,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        from aatm.llm.decisions import DecisionModel, has_probabilities

        if mode not in ("verify", "replace"):
            raise ValueError("mode must be 'verify' or 'replace'")
        if min_probability is not None and not 0.0 <= min_probability <= 1.0:
            raise ValueError("min_probability must be between 0 and 1, or None")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if isinstance(attributes, str):
            raise TypeError("attributes must be a list of names, not a string")
        if isinstance(model, str):
            from aatm.llm import create_decision_model

            model = create_decision_model(model, **dict(model_kwargs or {}))
        elif model_kwargs:
            raise ValueError("model_kwargs only applies when model is a model id")
        if not isinstance(model, DecisionModel):
            raise TypeError(
                "model must be a decision model id or a DecisionModel, got "
                f"{type(model).__name__}"
            )
        self.decision_model = model
        self.model_id: str = model.model_id
        self.mode: LabellerMode = mode
        self.attributes = list(attributes) if attributes is not None else None
        self.min_probability = min_probability
        self.instructions = instructions
        self.max_concurrency = max_concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None
        if min_probability is not None and not has_probabilities(model):
            logger.warning(
                "%s gives no probabilities: with min_probability set, every label "
                "it gives goes to review.",
                self.model_id,
            )

    # --- which attributes ---------------------------------------------------------

    def targets(
        self, definitions: EntityDefinitions
    ) -> dict[str, list[AttributeDefinition]]:
        """The labelled attributes of each entity key, in definition order.

        Raises:
            ValueError: If `attributes` names an unknown entity or attribute, or an
                entity without attributes.
        """
        if self.attributes is None:
            return {
                e.key: list(e.attributes) for e in definitions.entities if e.attributes
            }
        wanted: dict[str, set[str]] = {}
        for name in self.attributes:
            entity_key, _, attribute_key = name.partition(".")
            entity = definitions.get(entity_key)
            if entity is None:
                raise ValueError(f"Unknown entity {entity_key!r} in attributes")
            if not entity.attributes:
                raise ValueError(f"Entity {entity_key!r} has no attributes")
            keys = wanted.setdefault(entity_key, set())
            if not attribute_key:
                keys.update(a.key for a in entity.attributes)
            elif entity.get_attribute(attribute_key) is None:
                raise ValueError(
                    f"Entity {entity_key!r} has no attribute {attribute_key!r}"
                )
            else:
                keys.add(attribute_key)
        return {
            e.key: [a for a in e.attributes if a.key in wanted[e.key]]
            for e in definitions.entities
            if e.key in wanted
        }

    def fingerprint(self, definitions: EntityDefinitions) -> dict[str, Any]:
        """What the labels depend on, for resuming runs (`aatm.omop` runner)."""
        from aatm.llm.decisions import model_identity

        return {
            "labeller": type(self).__qualname__,
            "mode": self.mode,
            "min_probability": self.min_probability,
            "instructions": self.instructions,
            "attributes": {
                key: [{"key": a.key, "ordered": a.ordered} for a in attributes]
                for key, attributes in self.targets(definitions).items()
            },
            **model_identity(self.decision_model),
        }

    # --- replace mode: the extractor leaves the attributes out ---------------------

    def prompt_definitions(self, definitions: EntityDefinitions) -> EntityDefinitions:
        """The definitions the extractor uses: without the labelled attributes in
        ``replace`` mode, unchanged in ``verify`` mode."""
        if self.mode == "verify":
            return definitions
        targets = self.targets(definitions)
        entities = []
        for entity in definitions.entities:
            labelled = {a.key for a in targets.get(entity.key, [])}
            if labelled:
                # model_copy skips validation: the omop block may refer to the
                # attributes left out, and is not used by the extractor.
                entity = entity.model_copy(
                    update={
                        "attributes": [
                            a for a in entity.attributes if a.key not in labelled
                        ]
                    }
                )
            entities.append(entity)
        return definitions.model_copy(update={"entities": entities})

    def strip_candidates(
        self, candidates: ExtractedEntityCandidates, definitions: EntityDefinitions
    ) -> ExtractedEntityCandidates:
        """In ``replace`` mode, drop labels the extractor gave anyway."""
        if self.mode == "verify":
            return candidates
        targets = {
            key: {a.key for a in attributes}
            for key, attributes in self.targets(definitions).items()
        }
        entities = [
            c.model_copy(
                update={
                    "attributes": [
                        a
                        for a in c.attributes
                        if a.key.strip().casefold() not in targets.get(c.key, set())
                    ]
                }
            )
            for c in candidates.entities
        ]
        return candidates.model_copy(update={"entities": entities})

    # --- labelling ----------------------------------------------------------------

    async def decide(
        self,
        entity: EntityDefinition,
        attributes: Sequence[AttributeDefinition],
        mention: str,
        line: str,
    ) -> dict[str, AttributeDecision]:
        """Ask the attributes of one mention; return the answer of each, by key."""
        asked = [
            _AttributeQuestions(i, entity, a, self.instructions)
            for i, a in enumerate(attributes)
        ]
        questions = [q for a in asked for q in a.questions]
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        async with self._semaphore:
            decision = await self.decision_model.decide(
                {"entity": entity.name, "mention": mention, "line": line}, questions
            )
        return {a.attribute.key: a.read(decision) for a in asked}

    async def label(
        self,
        result: EntityValidationResult,
        segments: Sequence[TextSegment],
        definitions: EntityDefinitions,
    ) -> EntityValidationResult:
        """Label the attributes of each validated mention of `result`.

        Returns a new result: labels the model is sure of (and, in ``verify`` mode,
        agrees with) are kept, the others go to ``attribute_reviews``, and mentions
        left without a required attribute are rejected with their children.

        Raises:
            Exception: Any error of the decision model, once retries are spent.
        """
        targets = self.targets(definitions)
        lines = {s.id: s.text for s in segments}
        definitions_by_key = {e.key: e for e in definitions.entities}
        jobs = [
            (entity, definitions_by_key[entity.key], targets[entity.key])
            for entity in result.validated_entities
            if entity.key in targets
        ]
        decisions = await asyncio.gather(
            *(
                self.decide(
                    definition, attributes, entity.text, lines.get(entity.line, "")
                )
                for entity, definition, attributes in jobs
            )
        )
        result = result.model_copy(deep=True)
        by_id = {e.id: e for e in result.validated_entities}
        under_review: dict[str, set[str]] = {}
        for (entity, definition, attributes), answers in zip(jobs, decisions):
            kept = by_id[entity.id]
            for attribute in attributes:
                reason = self._apply(
                    kept, definition, attribute, answers[attribute.key], result
                )
                if reason is not None:
                    under_review.setdefault(kept.id, set()).add(attribute.key)
        return _reject_missing_required(result, targets, under_review, definitions)

    def _apply(
        self,
        entity: ExtractedEntity,
        definition: EntityDefinition,
        attribute: AttributeDefinition,
        answer: AttributeDecision,
        result: EntityValidationResult,
    ) -> Optional[AttributeReviewReason]:
        """Write one answer to `entity`, or send it to review; return the reason."""
        extractor_value = (
            entity.get_attribute(attribute.key) if self.mode == "verify" else None
        )
        reason: Optional[AttributeReviewReason] = None
        if answer.refused:
            reason = "refused"
        elif self.min_probability is not None and (
            answer.probability is None or answer.probability < self.min_probability
        ):
            reason = "low_probability"
        elif self.mode == "verify" and answer.label != extractor_value:
            reason = "disagreement"

        entity.attributes = [a for a in entity.attributes if a.key != attribute.key]
        entity.attribute_probabilities.pop(attribute.key, None)
        if reason is not None:
            result.attribute_reviews.append(
                AttributeReview(
                    entity_id=entity.id,
                    entity_key=entity.key,
                    text=entity.text,
                    attribute_key=attribute.key,
                    extractor_value=extractor_value,
                    decision_value=answer.label,
                    probability=answer.probability,
                    probabilities=answer.probabilities,
                    reason=reason,
                    model=self.model_id,
                )
            )
            return reason
        if answer.label is not None:
            entity.attributes.append(
                ExtractedAttribute(key=attribute.key, value=answer.label)
            )
            if answer.probability is not None:
                entity.attribute_probabilities[attribute.key] = answer.probability
        order = [a.key for a in definition.attributes]
        entity.attributes.sort(key=lambda a: order.index(a.key))
        return None


def _reject_missing_required(
    result: EntityValidationResult,
    targets: dict[str, list[AttributeDefinition]],
    under_review: dict[str, set[str]],
    definitions: EntityDefinitions,
) -> EntityValidationResult:
    """Reject mentions left without a labelled required attribute, and their
    children that no longer have a parent."""
    rejected: dict[str, list[RejectionReason]] = {}
    for entity in result.validated_entities:
        for attribute in targets.get(entity.key, []):
            if not attribute.required or entity.get_attribute(attribute.key):
                continue
            reasons = rejected.setdefault(entity.id, [])
            if attribute.key in under_review.get(entity.id, set()):
                reason: RejectionReason = "required_attribute_under_review"
            else:
                reason = "required_attribute_missing"
                result.required_attribute_issues.append(
                    AttributeIssue(
                        entity_id=entity.id,
                        entity_key=entity.key,
                        attribute_key=attribute.key,
                        reason="missing_required",
                        required=True,
                    )
                )
            if reason not in reasons:
                reasons.append(reason)
    if not rejected:
        return result

    parent_keys = definitions.parent_keys
    changed = True
    while changed:
        changed = False
        for entity in result.validated_entities:
            if entity.id in rejected or not entity.parents:
                continue
            parents = [p for p in entity.parents if p not in rejected]
            if len(parents) == len(entity.parents):
                continue
            entity.parents = parents
            if not parents and entity.key in parent_keys:
                rejected[entity.id] = ["parent_rejected"]
                changed = True

    kept: list[ExtractedEntity] = []
    for entity in result.validated_entities:
        if entity.id not in rejected:
            kept.append(entity)
            continue
        result.unmatched_entities.append(
            RejectedEntityCandidate(
                **entity.model_dump(exclude={"attribute_probabilities"}),
                reasons=rejected[entity.id],
            )
        )
    result.validated_entities = kept
    result.unmatched_keys = sorted(definitions.keys - {e.key for e in kept})
    return result
