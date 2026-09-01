# !/bin/bash

echo "Processing dataset1 Input KG"
uv run main.py preprocess --parameters configs/preprocess/ontomap/dataset1/dataset1_setting2_test/input_kg.yaml --cls OntomapStartingKG4FUNProcessor

echo "Processing dataset1 Schema"
uv run main.py preprocess --parameters configs/preprocess/ontomap/dataset1/dataset1_setting2_test/schema.yaml --cls OntomapSchemaKG4FUNProcessor

echo "Processing dataset1 Alignment"
uv run main.py preprocess --parameters configs/preprocess/ontomap/dataset1/dataset1_setting2_test/alignment.yaml --cls OntomapKG4FUNAlignmentProcessor