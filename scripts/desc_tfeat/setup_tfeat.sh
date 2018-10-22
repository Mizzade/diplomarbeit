#!/bin/bash
# Installs the python dependencies for tfeat descriptor.
# @INPUT:
#   $1: absolute path of the project's root directory.
# @OUTPUT
#   -

ROOT=$1
DIRNAME="$ROOT/desc_tfeat"

if [ -d $DIRNAME ]; then
  echo "Installing dependencies for TFeat Descriptor"
  cd $DIRNAME
  pipenv install
  cd $ROOT
  echo "Installation done."
fi
