#!/bin/bash
# Copies io_utils.py to target directory
# @INPUT:
#   $1: absolute path of the project's root directory.
#   $2: target dir name e.g. pipe_sift, desc_doap...
# @OUTPUT
#   -

ROOT=$1
TARGET_DIR=$2

echo "Copying io_utils to $ROOT/$TARGET_DIR"
  cd $ROOT/$TARGET_DIR
  cp $ROOT/extern/io_utils.py .
  cd $ROOT
echo "Copying done."
