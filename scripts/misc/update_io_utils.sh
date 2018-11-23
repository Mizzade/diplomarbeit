#!/bin/bash
# Copies the current version of io_utils.py to the corresponding model
# project directories.
# This is a helper function to quickly deploy a new version of io_utils.py
# @INPUT:
#   $1: absolute path of the project's root directory.
#   $2: {bool} - Rebuild TILDE's Dockerfile? Default: False
# @OUTPUT
#   None

ROOT=$1
REBUILD_DOCKERFILE=$2

if test -z ROOT ; then
  echo "Missing first parameter <ROOT>. Abort."
  exit 1
fi

target_dirs=("desc_doap" "desc_tfeat" "pipe_lift" "pipe_sift" "pipe_superpoint")
for i in "${target_dirs[@]}"
do
  $ROOT/scripts/misc/copy_io_utils.sh $ROOT $i
done

# If the second paramter is set, rebuild the Dockerfile.
if test "$2" ; then
  echo "Rebuilding TILDE Dockerfile."
  $ROOT/scripts/det_tilde/setup_docker_image_for_tilde.sh $ROOT '' true
fi


