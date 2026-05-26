# !/bin/bash

echo "Processing dataset1 Input KG"
uv run main.py preprocess --parameters configs/preprocess/ontomap/dataset1/input_kg.yaml --cls OntomapStartingKG4FUNProcessor

echo "Processing dataset1 Schema"
uv run main.py preprocess --parameters configs/preprocess/ontomap/dataset1/schema.yaml --cls OntomapSchemaKG4FUNProcessor