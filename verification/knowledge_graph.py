"""AI TrustLens - Structured Medical Knowledge Graph Module.

Provides a structured medical knowledge graph layer representing:
- Medical Entities: diseases, symptoms, medications, treatments, physiology, etiologies.
- Typed Relationships: causes, indicates, treats, contraindicates, defines_threshold, normal_range.
- Evidence Provenance: every relationship retains its exact source document, citation, and evidence passage.
- Verification Status: unverified vs established medical relationships.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RelationType:
    CAUSES = "causes"
    INDICATES = "indicates"
    TREATS = "treats"
    CONTRAINDICATES = "contraindicates"
    HAS_SYMPTOM = "has_symptom"
    DEFINES_THRESHOLD = "defines_threshold"
    NORMAL_RANGE = "normal_range"
    PREVENTS = "prevents"
    IS_A = "is_a"
    REGULATED_BY = "regulated_by"


class EntityCategory:
    DISEASE = "disease"
    SYMPTOM = "symptom"
    MEDICATION = "medication"
    TREATMENT = "treatment"
    PHYSIOLOGY = "physiology"
    CAUSE = "cause"
    ORGAN = "organ"
    OTHER = "other"


class MedicalEntity:
    """Represents a standardized medical concept."""

    def __init__(
        self,
        entity_id: str,
        name: str,
        category: str = EntityCategory.OTHER,
        aliases: Optional[Set[str]] = None,
        description: str = "",
    ):
        self.entity_id = entity_id
        self.name = name
        self.normalized_name = self.normalize_text(name)
        self.category = category.lower()
        self.aliases = {self.normalize_text(a) for a in (aliases or set())}
        self.aliases.add(self.normalized_name)
        self.description = description

    @staticmethod
    def normalize_text(text: str) -> str:
        t = re.sub(r"[^\w\s]", " ", text.lower())
        return re.sub(r"\s+", " ", t).strip()

    def matches(self, query: str) -> bool:
        norm_q = self.normalize_text(query)
        if norm_q in self.aliases or norm_q == self.normalized_name:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "category": self.category,
            "aliases": sorted(list(self.aliases)),
            "description": self.description,
        }


class MedicalRelation:
    """Represents a verified or unverified medical relationship triple with provenance."""

    def __init__(
        self,
        subject: str,
        predicate: str,
        object: str,
        source_id: str,
        citation: str,
        evidence_text: str,
        qualifiers: Optional[List[str]] = None,
        is_negated: bool = False,
        status: str = "established",  # "established", "unverified", "contradicted"
        confidence: float = 1.0,
    ):
        self.subject = subject.strip()
        self.predicate = predicate.strip().lower()
        self.object = object.strip()
        self.source_id = source_id
        self.citation = citation
        self.evidence_text = evidence_text
        self.qualifiers = qualifiers or []
        self.is_negated = is_negated
        self.status = status
        self.confidence = confidence

    @property
    def triple_key(self) -> str:
        neg = "NOT " if self.is_negated else ""
        return f"{self.subject} -> {neg}{self.predicate} -> {self.object}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "is_negated": self.is_negated,
            "qualifiers": self.qualifiers,
            "status": self.status,
            "confidence": round(self.confidence, 3),
            "source_id": self.source_id,
            "citation": self.citation,
            "evidence_text": self.evidence_text,
        }


class MedicalKnowledgeGraph:
    """In-memory medical knowledge graph with entity resolution and relation queries."""

    def __init__(self, load_defaults: bool = True):
        self.entities: Dict[str, MedicalEntity] = {}
        self.relations: List[MedicalRelation] = []

        if load_defaults:
            self._load_core_medical_kg()

    def add_entity(
        self,
        name: str,
        category: str = EntityCategory.OTHER,
        aliases: Optional[Set[str]] = None,
        description: str = "",
    ) -> MedicalEntity:
        """Register a medical entity in the graph."""
        norm = MedicalEntity.normalize_text(name)
        if norm in self.entities:
            existing = self.entities[norm]
            if aliases:
                existing.aliases.update({MedicalEntity.normalize_text(a) for a in aliases})
            return existing

        entity_id = f"ent::{category.lower()}::{norm.replace(' ', '_')}"
        entity = MedicalEntity(
            entity_id=entity_id,
            name=name,
            category=category,
            aliases=aliases,
            description=description,
        )
        self.entities[norm] = entity
        return entity

    def get_entity(self, name_or_alias: str) -> Optional[MedicalEntity]:
        """Find an entity by exact name or registered alias."""
        norm = MedicalEntity.normalize_text(name_or_alias)
        if norm in self.entities:
            return self.entities[norm]
        for ent in self.entities.values():
            if ent.matches(name_or_alias):
                return ent
        return None

    def add_relation(
        self,
        subject: str,
        predicate: str,
        object: str,
        source_id: str,
        citation: str,
        evidence_text: str,
        qualifiers: Optional[List[str]] = None,
        is_negated: bool = False,
        status: str = "established",
        confidence: float = 1.0,
    ) -> MedicalRelation:
        """Add a directed medical relationship edge to the knowledge graph."""
        rel = MedicalRelation(
            subject=subject,
            predicate=predicate,
            object=object,
            source_id=source_id,
            citation=citation,
            evidence_text=evidence_text,
            qualifiers=qualifiers,
            is_negated=is_negated,
            status=status,
            confidence=confidence,
        )
        self.relations.append(rel)
        return rel

    def query_relations(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        include_unverified: bool = False,
    ) -> List[MedicalRelation]:
        """Query knowledge graph edges by pattern matching on subject, predicate, or object."""
        matched: List[MedicalRelation] = []
        subj_norm = MedicalEntity.normalize_text(subject) if subject else None
        obj_norm = MedicalEntity.normalize_text(object) if object else None
        pred_norm = predicate.lower() if predicate else None

        for rel in self.relations:
            if not include_unverified and rel.status != "established":
                continue

            if pred_norm and rel.predicate != pred_norm:
                continue

            if subj_norm:
                r_subj_norm = MedicalEntity.normalize_text(rel.subject)
                ent = self.get_entity(rel.subject)
                subj_match = (
                    r_subj_norm == subj_norm
                    or subj_norm in r_subj_norm
                    or r_subj_norm in subj_norm
                    or (ent and ent.matches(subject or ""))
                )
                if not subj_match:
                    continue

            if obj_norm:
                r_obj_norm = MedicalEntity.normalize_text(rel.object)
                ent_obj = self.get_entity(rel.object)
                obj_match = (
                    r_obj_norm == obj_norm
                    or obj_norm in r_obj_norm
                    or r_obj_norm in obj_norm
                    or (ent_obj and ent_obj.matches(object or ""))
                )
                if not obj_match:
                    continue

            matched.append(rel)

        return matched

    def check_assertion(
        self,
        subject: str,
        predicate: str,
        object: str,
        is_negated: bool = False,
    ) -> Tuple[bool, bool, List[MedicalRelation], str]:
        """Check whether a medical assertion is corroborated or contradicted by the KG.

        Returns (is_supported, is_contradicted, matched_relations, explanation).
        """
        # Look for identical or related relations
        relations = self.query_relations(subject=subject, predicate=predicate)
        obj_norm = MedicalEntity.normalize_text(object)

        supporting: List[MedicalRelation] = []
        contradicting: List[MedicalRelation] = []

        for rel in relations:
            rel_obj_norm = MedicalEntity.normalize_text(rel.object)
            is_same_object = (
                rel_obj_norm == obj_norm
                or obj_norm in rel_obj_norm
                or rel_obj_norm in obj_norm
            )

            # Numerical / threshold check (e.g. 100.4°F vs 104°F)
            if predicate in (RelationType.DEFINES_THRESHOLD, RelationType.NORMAL_RANGE):
                from .verifier import _check_numeric_contradiction, _extract_numbers_and_units
                c_nums = _extract_numbers_and_units(object)
                r_nums = _extract_numbers_and_units(rel.object)
                num_conflict = _check_numeric_contradiction(c_nums, r_nums, set(), set())
                if num_conflict:
                    contradicting.append(rel)
                elif c_nums and r_nums:
                    supporting.append(rel)
                elif is_same_object and (rel.is_negated == is_negated):
                    supporting.append(rel)
                else:
                    contradicting.append(rel)
                continue

            # Directional / Contraindication check
            if predicate == RelationType.TREATS and rel.predicate == RelationType.CONTRAINDICATES:
                if is_same_object:
                    contradicting.append(rel)
                    continue

            if is_same_object:
                if rel.is_negated == is_negated:
                    supporting.append(rel)
                else:
                    contradicting.append(rel)

        if supporting and not contradicting:
            first = supporting[0]
            expl = f"Corroborated by Knowledge Graph: {first.triple_key} (Source: {first.citation})."
            return True, False, supporting, expl

        if contradicting and not supporting:
            first = contradicting[0]
            expl = f"Contradicted by Knowledge Graph: established relation is {first.triple_key} (Source: {first.citation})."
            return False, True, contradicting, expl

        if supporting and contradicting:
            expl = "Conflicting evidence in Knowledge Graph across distinct medical relationships."
            return False, False, supporting + contradicting, expl

        return False, False, [], "No direct structured relationship found in Knowledge Graph."

    def find_connected_knowledge(self, claim_text: str) -> List[Dict[str, Any]]:
        """Extract structured KG context relevant to any entities mentioned in a claim."""
        claim_norm = MedicalEntity.normalize_text(claim_text)
        relevant_relations: List[MedicalRelation] = []

        # Find any registered entities mentioned in the claim
        for ent in self.entities.values():
            if ent.normalized_name in claim_norm or any(a in claim_norm for a in ent.aliases if len(a) > 3):
                rels = self.query_relations(subject=ent.name)
                rels_obj = self.query_relations(object=ent.name)
                relevant_relations.extend(rels)
                relevant_relations.extend(rels_obj)

        # Deduplicate by triple key
        seen: Set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for r in relevant_relations:
            if r.triple_key not in seen:
                seen.add(r.triple_key)
                deduped.append(r.to_dict())

        return deduped

    def _load_core_medical_kg(self) -> None:
        """Populate the core medical knowledge graph with evidence-grounded clinical facts."""
        # 1. Register Core Entities
        fever = self.add_entity("Fever", EntityCategory.SYMPTOM, {"pyrexia", "febrile response", "high temperature"})
        norm_temp = self.add_entity("Normal Body Temperature", EntityCategory.PHYSIOLOGY, {"baseline temperature"})
        infection = self.add_entity("Infection", EntityCategory.CAUSE, {"viral infection", "bacterial infection", "infections"})
        inflammation = self.add_entity("Inflammation", EntityCategory.CAUSE, {"inflammatory disease", "autoimmune"})
        cancer = self.add_entity("Cancer", EntityCategory.CAUSE, {"malignancy", "neoplasm"})
        vaccine = self.add_entity("Vaccination", EntityCategory.CAUSE, {"vaccines", "immunization"})
        heat_exhaustion = self.add_entity("Heat Exhaustion", EntityCategory.CAUSE, {"environmental heat"})
        aspirin = self.add_entity("Aspirin", EntityCategory.MEDICATION, {"acetylsalicylic acid"})
        pediatric_viral = self.add_entity("Pediatric Viral Infection", EntityCategory.DISEASE, {"children with viral illness", "viral fever in children"})
        htn = self.add_entity("High Blood Pressure", EntityCategory.DISEASE, {"hypertension", "elevated bp"})
        paracetamol = self.add_entity("Paracetamol", EntityCategory.MEDICATION, {"acetaminophen", "tylenol"})
        ibuprofen = self.add_entity("Ibuprofen", EntityCategory.MEDICATION, {"advil", "motrin"})
        diabetes_t1 = self.add_entity("Type 1 Diabetes", EntityCategory.DISEASE, {"juvenile diabetes", "t1d"})
        green_tea = self.add_entity("Green Tea", EntityCategory.TREATMENT, {"herbal tea"})

        # 2. Add Grounded Triples with Full Provenance
        # Fever Threshold
        self.add_relation(
            subject="Fever",
            predicate=RelationType.DEFINES_THRESHOLD,
            object="100.4°F (38°C) or higher",
            source_id="doc_cdc_temp_2024",
            citation="CDC > Body Temperature Assessment",
            evidence_text="A fever is generally considered to be a measured body temperature of 100.4°F (38°C) or higher.",
            status="established",
        )

        # Normal Temperature Range
        self.add_relation(
            subject="Normal Body Temperature",
            predicate=RelationType.NORMAL_RANGE,
            object="97°F (36.1°C) to 99°F (37.2°C)",
            source_id="doc_nih_temp_2024",
            citation="NIH MedlinePlus > Body Temperature Regulation",
            evidence_text="Normal body temperature is typically between 97°F (36.1°C) and 99°F (37.2°C).",
            status="established",
        )

        # Causes of Fever
        self.add_relation(
            subject="Infection",
            predicate=RelationType.CAUSES,
            object="Fever",
            source_id="doc_who_fever_2023",
            citation="WHO > Fever Clinical Overview",
            evidence_text="Common causes of fever include viral, bacterial, fungal, or parasitic infections.",
            qualifiers=["common"],
            status="established",
        )
        self.add_relation(
            subject="Inflammation",
            predicate=RelationType.CAUSES,
            object="Fever",
            source_id="doc_who_fever_2023",
            citation="WHO > Fever Clinical Overview",
            evidence_text="Inflammation such as from autoimmune diseases is a common cause of fever.",
            qualifiers=["common"],
            status="established",
        )
        self.add_relation(
            subject="Cancer",
            predicate=RelationType.CAUSES,
            object="Fever",
            source_id="doc_who_fever_2023",
            citation="WHO > Fever Clinical Overview",
            evidence_text="Cancer in some cases causes fever.",
            qualifiers=["in some cases"],
            status="established",
        )
        self.add_relation(
            subject="Vaccination",
            predicate=RelationType.CAUSES,
            object="Fever",
            source_id="doc_who_fever_2023",
            citation="WHO > Fever Clinical Overview",
            evidence_text="Vaccinations are a recognized cause of fever.",
            status="established",
        )
        self.add_relation(
            subject="Heat Exhaustion",
            predicate=RelationType.CAUSES,
            object="Fever",
            source_id="doc_who_fever_2023",
            citation="WHO > Fever Clinical Overview",
            evidence_text="Extreme environmental heat can lead to heat exhaustion and fever.",
            status="established",
        )

        # Aspirin Contraindication
        self.add_relation(
            subject="Aspirin",
            predicate=RelationType.CONTRAINDICATES,
            object="Pediatric Viral Infection",
            source_id="doc_fda_aspirin_2023",
            citation="FDA > Pediatric Safety Warning: Aspirin",
            evidence_text="Aspirin is contraindicated in children and teenagers who have or are recovering from viral infections due to Reye's syndrome.",
            qualifiers=["severe risk of Reye's syndrome"],
            is_negated=False,
            status="established",
        )

        # Treatments for Fever
        self.add_relation(
            subject="Paracetamol",
            predicate=RelationType.TREATS,
            object="Fever",
            source_id="doc_cdc_temp_2024",
            citation="CDC > Temperature Assessment",
            evidence_text="Paracetamol helps relieve discomfort associated with fever.",
            status="established",
        )
        self.add_relation(
            subject="Ibuprofen",
            predicate=RelationType.TREATS,
            object="Fever",
            source_id="doc_cdc_temp_2024",
            citation="CDC > Temperature Assessment",
            evidence_text="Ibuprofen helps relieve discomfort associated with fever.",
            status="established",
        )

        # Hypertension Symptoms
        self.add_relation(
            subject="High Blood Pressure",
            predicate=RelationType.HAS_SYMPTOM,
            object="Asymptomatic / No Symptoms",
            source_id="doc_aha_htn_2024",
            citation="AHA > High Blood Pressure Symptoms",
            evidence_text="High blood pressure usually has no symptoms and is often called the silent killer.",
            qualifiers=["usually"],
            status="established",
        )

        # Type 1 Diabetes and Green Tea (Negative)
        self.add_relation(
            subject="Green Tea",
            predicate=RelationType.TREATS,
            object="Type 1 Diabetes",
            source_id="doc_who_diabetes_2023",
            citation="WHO > Diabetes Mellitus Overview",
            evidence_text="Type 1 diabetes is not curable with dietary supplements, green tea, or herbal remedies.",
            is_negated=True,
            status="established",
        )
