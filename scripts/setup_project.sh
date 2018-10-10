#!/bin/bash
ROOT=$(pwd)

# TILDE
echo "Starting setup for TILDE..."
$ROOT/scripts/create_docker_image_for_tilde.sh $ROOT
