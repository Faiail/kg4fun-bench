# !/bin/bash

echo "Processing dataset2 Input KG"
uv run main.py preprocess --parameters configs/preprocess/ontomap/dataset2/input_kg.yaml --cls OntomapStartingKG4FUNProcessor

echo "Processing dataset2 Schema"
uv run main.py preprocess --parameters configs/preprocess/ontomap/dataset2/schema.yaml --cls OntomapSchemaKG4FUNProcessor