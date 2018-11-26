#!/bin/bash
# Checks if the docker tilde_opencv-3.4.1_python-3.6.6:latest exists.
# If not, create it.
# INPUT
#   ROOT {str}              : Absolute path of the root of the repository.
#   DOCKER_IMAGE_NAME {str} : Name of the docker image containing TILDE. If not
#                             not set, default to "tilde_opencv-3.4.1_python-3.6.6:latest".
#   FORCE_BUILD {bool}      : Force rebuild whether or not the image was found.
# OUTPUT
#   None

ROOT=$1
DOCKER_IMAGE_NAME=$2
FORCE_BUILD=$3

if [ !DOCKER_IMAGE_NAME ]; then
  DOCKER_IMAGE_NAME="tilde_opencv-3.4.1_python-3.6.6:latest"
fi

echo "In: setup_docker_image_for_tilde.sh"
if test ! -z "$(docker images -q $DOCKER_IMAGE_NAME)"; then
  echo "$DOCKER_IMAGE_NAME exists."
else
  echo "$DOCKER_IMAGE_NAME does not exist. Creating it."
  nvidia-docker build -t $DOCKER_IMAGE_NAME -f "$ROOT/det_tilde/Dockerfile" .
fi

# If FORCE_BUILD parameter is set, rebuild the docker image.
if test -z $3 ; then
  echo ''
else
  echo "Force building of Dockerfile."
  nvidia-docker build -t $DOCKER_IMAGE_NAME -f "$ROOT/det_tilde/Dockerfile" .
fi
