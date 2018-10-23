#!/bin/bash
# Installs the python dependencies for superpoint pipeline.
# @INPUT:
#   $1: absolute path of the project's root directory.
# @OUTPUT
#   -

ROOT=$1
DIRNAME="$ROOT/pipe_superpoint"

if [ -d $DIRNAME ]; then
  echo "Installing dependencies for SuperPoint pipeline."
  cd $DIRNAME
  pipenv install
  cd $ROOT
  echo "Installation done."
fi
