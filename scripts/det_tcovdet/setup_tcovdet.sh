#!/bin/bash
# Installs the python dependencies for tcovdet detector.
# @INPUT:
#   $1: absolute path of the project's root directory.
# @OUTPUT
#   -

ROOT=$1
DIRNAME="$ROOT/det_tcovdet"

if [ -d $DIRNAME ]; then
  echo "Installing dependencies for TCovDet detector."
  cd $DIRNAME
  pipenv install
  cd $ROOT
  echo "Installation done."
fi
