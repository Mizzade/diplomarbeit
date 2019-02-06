#!/bin/bash
# Installs the python dependencies for DOAP descriptor.
# @INPUT:
#   $1: absolute path of the project's root directory.
# @OUTPUT
#   -

ROOT=$1
DIRNAME="$ROOT/desc_doap"

if [ -d $DIRNAME ]; then
  echo "Installing dependencies for DOAP descriptor."
  cd $DIRNAME
  pipenv install
  cd $ROOT
  echo "Installation done."
fi
