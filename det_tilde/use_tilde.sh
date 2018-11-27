#!/bin/bash
# Calls the docker container running the TILDE detector and performs it
# Input:
#   MOUNT_IN {str}   : Path to the image directory containing subsets on the
#                      HOST system.
#   MOUNT_OUT {str}  : Path to the output directory where results of the models
#                      is stored.
#   TMP_DIR {str}    : Path to the tmp directory containing the config file for
#                      the TILDE detector.
#   CONFIG_FILE {str}: Name of the config file inside the TMP_DIR.
#   DOCKER_IMAGE_NAME {str} : Name of the docker image to use. Defaults to
#                             tilde_opencv-3.4.1_python-3.6.6:latest

MOUNT_IN=$1
MOUNT_OUT=$2
TMP_DIR=$3
CONFIG_FILE=$4
DOCKER_IMAGE_NAME=$5

if [ !DOCKER_IMAGE_NAME ]; then
  DOCKER_IMAGE_NAME="tilde_opencv-3.4.1_python-3.6.6:latest"
fi

docker run \
--rm \
--mount type=bind,source="$MOUNT_IN",target="$MOUNT_IN" \
--mount type=bind,source="$MOUNT_OUT",target="$MOUNT_OUT" \
--mount type=bind,source="$TMP_DIR",target="$TMP_DIR" \
$(docker images -q $DOCKER_IMAGE_NAME) \
python use_tilde.py $CONFIG_FILES
