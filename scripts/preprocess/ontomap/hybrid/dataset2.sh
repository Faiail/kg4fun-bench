# !/bin/bash

echo "Processing dataset2 Input KG"
uv run main.py preprocess --parameters configs/preprocess/ontomap/dataset2/dataset2_hybrid_test/input_kg.yaml --cls OntomapStartingKG4FUNProcessor

echo "Processing dataset2 Schema"
uv run main.py preprocess --parameters configs/preprocess/ontomap/dataset2/dataset2_hybrid_test/schema.yaml --cls OntomapSchemaKG4FUNProcessor

echo "Processing dataset2 Alignment"
uv run main.py preprocess --parameters configs/preprocess/ontomap/dataset2/dataset2_hybrid_test/alignment.yaml --cls OntomapKG4FUNAlignmentProcessor