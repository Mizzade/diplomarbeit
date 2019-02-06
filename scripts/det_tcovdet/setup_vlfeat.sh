#!/bin/bash
# Downloads and extracts vlfeat 0.9.21 from http://www.vlfeat.org/download/vlfeat-0.9.21.tar.gz

ROOT=$1
FILENAME="vlfeat-0.9.21"
TARNAME="$FILENAME.tar.gz"
MODULE="det_tcovdet"

if [ ! -f "$ROOT/extern/$TARNAME" ]; then
  echo "Downloading $FILENAME."
  wget -nd "http://www.vlfeat.org/download/$TARNAME" -P "$ROOT/$MODULE"
  echo "Download done."
else
  echo "Copying $FILENAME."
  cp "$ROOT/extern/$TARNAME" "$ROOT/$MODULE"
  echo "Copying done."
fi

if [ -f "$ROOT/$MODULE/$TARNAME" ]; then
  echo "Extracting $FILENAME."
  tar -zxvf "$ROOT/$MODULE/$TARNAME" -C "$ROOT/$MODULE/"
  echo "Extraction done."
fi

if [ -d "$ROOT/$MODULE/$FILENAME" ]; then
  echo "Compiling $FILENAME."
  cd -P "$ROOT/$MODULE/$FILENAME"
  make ARCH=glnxa64 MEX=$MATLAB_HOME/mex
  echo "Compilation done."

  echo "Copying mex folder."
  cp -r "$ROOT/extern/mex" "$ROOT/$MODULE/$FILENAME/toolbox"
  echo "Copying mex done."
fi

if [ -f "$ROOT/$MODULE/$TARNAME" ]; then
  echo "Removing $TARNAME."
  rm "$ROOT/$MODULE/$TARNAME"
  echo "Removal done."
fi
