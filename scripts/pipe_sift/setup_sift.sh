#!/bin/bash
# Installs the python dependencies for sift pipeline.
# @INPUT:
#   $1: absolute path of the project's root directory.
# @OUTPUT
#   -

ROOT=$1
DIRNAME="$ROOT/pipe_sift"

if [ -d $DIRNAME ]; then
  echo "Installing dependencies for SIFT pipeline."
  cd $DIRNAME
  pipenv install
  cd $ROOT
  echo "Installation done."
fi
