#!/bin/bash
# Checks if the docker image tilde_opencv-2.4.9_python-3.6.6 exists.
# If not, create it.

ROOT=$1
DOCKER_IMAGE_NAME="tilde_opencv-3.4.1_python-3.6.6:latest"

echo "In: setup_docker_image_for_tilde.sh"
if test ! -z "$(docker images -q $DOCKER_IMAGE_NAME)"; then
  echo "$DOCKER_IMAGE_NAME exists."
else
  echo "$DOCKER_IMAGE_NAME does not exist. Creating it."
  nvidia-docker build -t $DOCKER_IMAGE_NAME -f "$ROOT/det_tilde/Dockerfile" .
fi
