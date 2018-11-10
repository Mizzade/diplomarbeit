#!/bin/bash
# Calls the docker container running the TILDE detector and performs it
# Input:
#   MOUNT_IN {str}   : Path to the image directory containing subsets on the
#                      HOST system.
#   MOUNT_OUT {str}  : Path to the output directory where results of the models
#                      is stored.
#   FILE_LIST: {str} : List of file paths to images, to be used with TILDE.
MOUNT_IN=$1
MOUNT_OUT=$2
FILE_LIST=$3

#echo "Starting USE_TILDE.sh with MOUNT_IN: $MOUNT_IN, MOUNT_OUT: $MOUNT_OUT and FILE_LIST: $FILE_LIST"
docker exec 415e1e1a4ccd python use_tilde.py $MOUNT_OUT $FILE_LIST
#docker exec 415e1e1a4ccd python hello_world2.py $FILE_LIST

# docker run \
# --rm \
# --mount type=bind,source="$MOUNT_IN",target="$MOUNT_IN" \
# --mount type=bind,source="$MOUNT_OUT",target="$MOUNT_OUT" \
# $(docker images -q tilde_opencv-3.4.1_python-3.6.6:latest) \
# python use_tilde.py $MOUNT_OUT $FILE_LIST


#./hello_world.sh $MOUNT_IN
#python hello_world2.py $MOUNT_IN $MOUNT_OUT $FILE_LIST

#./use_tilde.py $MOUNT_OUT $FILE_LIST
