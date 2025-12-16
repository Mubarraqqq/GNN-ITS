# ontology_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from owlready2 import get_ontology, Thing

ONTO_PATH = "ont.rdf"  # adjust if needed

class OntologyEngine:


    def __init__(self, path: str = ONTO_PATH):
        self.onto = get_ontology(path).load()

        # Namespaces (co-ode for properties/individuals, gnn-its for core classes)
        self.O = self.onto.get_namespace("http://www.co-ode.org/ontologies/ont.owl#")
        self.G = self.onto.get_namespace("http://www.example.org/gnn-its#")





@dataclass
class ObjectiveInfo:
    iri: str
    name: str
    description: Optional[str]
    level: Optional[str]


@dataclass
class TaskInfo:
    iri: str
    name: str
    type_name: str
    description: Optional[str]
    difficulty: Optional[str]
    estimated_time: Optional[str]
    requires_coding: bool
    concept_iris: List[str]
    graph_iris: List[str]
    dataset_iris: List[str]


@dataclass
class AssessmentInfo:
    iri: str
    name: str
    description: Optional[str]
    current_score: Optional[float]
    max_score: Optional[float]
    passing_score: Optional[float]
    required_concepts: List[str]

    # ---------- utility ----------

    @staticmethod
    def _get_single(obj: Thing, attr: str) -> Optional[Any]:
        """Return first value of list-valued attribute or None."""
        vals = getattr(obj, attr, [])
        if not vals:
            return None
        return vals[0]

    @staticmethod
    def _get_all(obj: Thing, attr: str) -> List[Any]:
        return list(getattr(obj, attr, []))

    @staticmethod
    def _label(obj: Thing) -> str:
        # owlready2 usually exposes `name`; rdfs:label could also be used.
        return getattr(obj, "name", obj.iri)

    # ---------- objectives ----------

    def list_objectives(self) -> List[ObjectiveInfo]:
        objs = list(self.G.LearningObjective.instances())
        return [self.objective_info(o) for o in objs]

    def objective_info(self, obj: Thing) -> ObjectiveInfo:
        return ObjectiveInfo(
            iri=obj.iri,
            name=self._label(obj),
            description=self._get_single(obj, "objectiveDescription"),
            level=self._get_single(obj, "objectiveLevel"),
        )

    def get_objective_by_iri(self, iri: str) -> Optional[Thing]:
        return self.onto.search_one(iri=iri)

    # ---------- tasks / learning activities ----------

    def tasks_for_objective(self, obj: Thing) -> List[TaskInfo]:
        tasks = []
        for t in self.G.LearningTask.instances():
            if obj in list(getattr(t, "targetsObjective", [])):
                tasks.append(self.task_info(t))
        return tasks

    def task_info(self, t: Thing) -> TaskInfo:
        # class name for type: e.g. ConceptExplanation, WorkedExample...
        type_name = t.is_a[0].name if t.is_a else "LearningTask"

        return TaskInfo(
            iri=t.iri,
            name=self._label(t),
            type_name=type_name,
            description=self._get_single(t, "taskDescription"),
            difficulty=self._get_single(t, "taskDifficultyLevel"),
            estimated_time=self._get_single(t, "estimatedCompletionTime"),
            requires_coding=bool(self._get_single(t, "requiresCoding") or False),
            concept_iris=[c.iri for c in self._get_all(t, "teachesConcept")],
            graph_iris=[g.iri for g in self._get_all(t, "producesGraph")],
            dataset_iris=[d.iri for d in self._get_all(t, "usesGraphDataset")],
        )

    # ---------- assessments ----------

    def assessments_for_objective(self, obj: Thing) -> List[AssessmentInfo]:
        assessments = []
        for a in self.G.Assessment.instances():
            if obj in list(getattr(a, "assessesObjective", [])):
                assessments.append(self.assessment_info(a))
        return assessments

    def assessment_info(self, a: Thing) -> AssessmentInfo:
        return AssessmentInfo(
            iri=a.iri,
            name=self._label(a),
            description=self._get_single(a, "assessmentDescription"),
            current_score=self._get_single(a, "currentScore"),
            max_score=self._get_single(a, "maxScore"),
            passing_score=self._get_single(a, "passingScore"),
            required_concepts=[c.iri for c in self._get_all(a, "requiresConcept")],
        )

    # ---------- descriptive helpers for UI ----------

    def describe_concept(self, iri: str) -> Dict[str, Any]:
        c = self.onto.search_one(iri=iri)
        if not c:
            return {"iri": iri, "name": iri, "kind": "Unknown", "details": {}}

        # kind: GNNConcept / GraphDataset / GraphType / GraphInstance / LossFunction / AccuracyMetric, etc.
        kind = "Thing"
        for cls in (
            self.G.GNNConcept,
            self.G.GraphDataset,
            self.G.GraphType,
            self.G.GraphInstance,
            self.G.LossFunction,
            self.G.AccuracyMetric,
        ):
            if isinstance(c, cls):
                kind = cls.name
                break

        details = {}
        # GraphDataset meta-data, if present
        if isinstance(c, self.G.GraphDataset):
            details["datasetName"] = self._get_single(c, "datasetName")
            details["numGraphs"] = self._get_single(c, "numGraphs")
            details["numNodeFeatures"] = self._get_single(c, "numNodeFeatures")
            details["sourceURL"] = self._get_single(c, "sourceURL")

        # GraphInstance meta-data
        if isinstance(c, self.G.GraphInstance):
            details["graphLabel"] = self._get_single(c, "graphLabel")
            details["matrixSize"] = self._get_single(c, "matrixSize")
            details["numNodes"] = self._get_single(c, "numNodes")
            details["numEdges"] = self._get_single(c, "numEdges")

        return {
            "iri": c.iri,
            "name": self._label(c),
            "kind": kind,
            "details": details,
        }
