#!/bin/bash
# Calls the docker container running the TILDE detector and performs it
# Input:
#   MOUNT_IN {str}   : Path to the image directory containing subsets on the
#                      HOST system.
#   MOUNT_OUT {str}  : Path to the output directory where results of the models
#                      is stored.
#   FILE_LIST: {str} : List of file paths to images, to be used with TILDE.
#   DOCKER_IMAGE_NAME {str} : Name of the docker image to use. Defaults to
#                             tilde_opencv-3.4.1_python-3.6.6:latest

MOUNT_IN=$1
MOUNT_OUT=$2
FILE_LIST=$3
DOCKER_IMAGE_NAME=$4

if [ !DOCKER_IMAGE_NAME ]; then
  DOCKER_IMAGE_NAME="tilde_opencv-3.4.1_python-3.6.6:latest"
fi

#echo "Starting USE_TILDE.sh with MOUNT_IN: $MOUNT_IN, MOUNT_OUT: $MOUNT_OUT and FILE_LIST: $FILE_LIST"

docker run \
--rm \
--mount type=bind,source="$MOUNT_IN",target="$MOUNT_IN" \
--mount type=bind,source="$MOUNT_OUT",target="$MOUNT_OUT" \
$(docker images -q $DOCKER_IMAGE_NAME) \
python use_tilde.py $MOUNT_OUT $FILE_LIST
