#!/bin/bash
# Call doap over matlab from bash script.
# Uses doap to find each .csv file containing patches in the location
# of <INPUT_DIR> uses DOAP to transform them into descriptors.
# Write the descriptors to <OUTPUT_DIR> and keep the general structure
# of <INPUT_DIR>.

ROOT=$1
DOAP_DIR="$ROOT/desc_doap"
EXAMPLE_DIR="$ROOT/examples/desc_doap"
VLFEAT_ROOT="$DOAP_DIR/vlfeat-0.9.21"
MATCONVNET_ROOT="$DOAP_DIR/matconvnet-1.0-beta25"
DOAP_MODEL="$DOAP_DIR/HPatches_ST_LM_128d.mat"
LAYERS_DIR="$DOAP_DIR"
INPUT_DIR="$EXAMPLE_DIR/test_in"
OUTPUT_DIR="$EXAMPLE_DIR/test_out"

cd -P "$DOAP_DIR"
matlab -r "use_doap_with_csv('$VLFEAT_ROOT','$MATCONVNET_ROOT', '$DOAP_MODEL', '$LAYERS_DIR', '$INPUT_DIR', '$OUTPUT_DIR'); exit(0)";

