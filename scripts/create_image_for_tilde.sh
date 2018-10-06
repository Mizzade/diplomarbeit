#!/bin/bash
# Checks if the docker image tilde_opencv-2.4.9_python-3.6.6 exists.
# If not, create it.

ROOT=$1

echo "In: create_image_for_tilde.sh"
if test ! -z "$(docker images -q tilde_opencv-2.4.9_python-3.6.6:latesst)"; then
  echo "tilde_opencv-2.4.9_python-3.6.6:latest exists."
else
  echo "tilde_opencv-2.4.9_python-3.6.6:latest does not exist. Creating it."
  nvidia-docker build -t tilde_opencv-2.4.9_python-3.6.6 -f "$ROOT/docker/tilde/Dockerfile" .
fi
