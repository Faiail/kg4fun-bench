# -*- coding: utf-8 -*-
import os

from src.models.ontomap.base import (
    KG4FunAlignments,
    KG4FunOMDataset,
    KG4FunOntology,
)

track = "kg4fun"


class KG4FunDataset1OMDataset(KG4FunOMDataset):
    track = track
    ontology_name = "dataset1"
    source_ontology = KG4FunOntology(
        ontology_file_name="input_kg/processed_ontology.json"
    )
    target_ontology = KG4FunOntology(
        ontology_file_name="schema/processed_ontology.json"
    )
    alignments = KG4FunAlignments(
        reference_file_name="alignment/alignment.json"
    )
    working_dir = os.path.join(track, ontology_name)


class KG4FunDataset2OMDataset(KG4FunOMDataset):
    track = track
    ontology_name = "dataset2"
    source_ontology = KG4FunOntology(
        ontology_file_name="input_kg/processed_ontology.json"
    )
    target_ontology = KG4FunOntology(
        ontology_file_name="schema/processed_ontology.json"
    )
    alignments = KG4FunAlignments(
        reference_file_name="alignment/alignment.json"
    )
    working_dir = os.path.join(track, ontology_name)


class KG4FunDataset1HybridTestOMDataset(KG4FunOMDataset):
    track = track
    ontology_name = "dataset1_hybrid_test"
    source_ontology = KG4FunOntology(
        ontology_file_name="input_kg/processed_ontology.json"
    )
    target_ontology = KG4FunOntology(
        ontology_file_name="schema/processed_ontology.json"
    )
    alignments = KG4FunAlignments(
        reference_file_name="alignment/alignment.json"
    )
    working_dir = os.path.join(track, ontology_name)


class KG4FunDataset1InductiveTestOMDataset(KG4FunOMDataset):
    track = track
    ontology_name = "dataset1_inductive_test"
    source_ontology = KG4FunOntology(
        ontology_file_name="input_kg/processed_ontology.json"
    )
    target_ontology = KG4FunOntology(
        ontology_file_name="schema/processed_ontology.json"
    )
    alignments = KG4FunAlignments(
        reference_file_name="alignment/alignment.json"
    )
    working_dir = os.path.join(track, ontology_name)
class KG4FunDataset2HybridTestOMDataset(KG4FunOMDataset):
    track = track
    ontology_name = "dataset2_hybrid_test"
    source_ontology = KG4FunOntology(
        ontology_file_name="input_kg/processed_ontology.json"
    )
    target_ontology = KG4FunOntology(
        ontology_file_name="schema/processed_ontology.json"
    )
    alignments = KG4FunAlignments(
        reference_file_name="alignment/alignment.json"
    )

class KG4FunDataset2InductiveTestOMDataset(KG4FunOMDataset):
    track = track
    ontology_name = "dataset2_inductive_test"
    source_ontology = KG4FunOntology(
        ontology_file_name="input_kg/processed_ontology.json"
    )
    target_ontology = KG4FunOntology(
        ontology_file_name="schema/processed_ontology.json"
    )
    alignments = KG4FunAlignments(
        reference_file_name="alignment/alignment.json"
    )
