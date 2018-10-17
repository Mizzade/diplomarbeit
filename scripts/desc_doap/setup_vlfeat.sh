#!/bin/bash
# Downloads and extracts vlfeat 0.9.21 from http://www.vlfeat.org/download/vlfeat-0.9.21.tar.gz

ROOT=$1
FILENAME="vlfeat-0.9.21"
TARNAME="$FILENAME.tar.gz"

if [ ! -f "$ROOT/extern/$TARNAME" ]; then
  echo "Downloading $FILENAME."
  wget -nd "http://www.vlfeat.org/download/$TARNAME" -P "$ROOT/desc_doap"
  echo "Download done."
else
  echo "Copying $FILENAME."
  cp "$ROOT/extern/$TARNAME" "$ROOT/desc_doap"
  echo "Copying done."
fi

if [ -f "$ROOT/desc_doap/$TARNAME" ]; then
  echo "Extracting $FILENAME."
  tar -zxvf "$ROOT/desc_doap/$TARNAME" -C "$ROOT/desc_doap/"
  echo "Extraction done."
fi

if [ -d "$ROOT/desc_doap/$FILENAME" ]; then
  echo "Compiling $FILENAME."
  cd -P "$ROOT/desc_doap/$FILENAME"
  make ARCH=glnxa64 MEX=$MATLAB_HOME/mex
  echo "Compilation done."

  echo "Copying mex folder."
  cp -r "$ROOT/extern/mex" "$ROOT/desc_doap/$FILENAME/toolbox"
  echo "Copying mex done."
fi


if [ -f "$ROOT/desc_doap/$TARNAME" ]; then
  echo "Removing $TARNAME."
  rm "$ROOT/desc_doap/$TARNAME"
  echo "Removal done."
fi
