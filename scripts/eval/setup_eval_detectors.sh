#!/bin/bash
# Installs the python dependencies evaluation of detectors.
# @INPUT:
#   $1: absolute path of the project's root directory.
# @OUTPUT
#   -

ROOT=$1
DIRNAME="$ROOT/evaluation/detectors"

if [ -d $DIRNAME ]; then
  echo "Installing dependencies for evaluation of detectors."
  cd $DIRNAME
  pipenv install
  cd $ROOT
  echo "Installation done."
fi
