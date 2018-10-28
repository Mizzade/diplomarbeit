#!/bin/bash
# Installs the python dependencies for LIFT pipeline.
# @INPUT:
#   $1: absolute path of the project's root directory.
# @OUTPUT
#   -

ROOT=$1
DIRNAME="$ROOT/pipe_lift"

if [ -d $DIRNAME ]; then
  echo "Installing dependencies for LIFT pipeline."
  cd $DIRNAME
  pip install -r requirements.txt
  cd $ROOT
  echo "Installation done."
fi
