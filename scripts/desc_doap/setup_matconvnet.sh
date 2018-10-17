#!/bin/bash
# Downloads and extracts matconvnet-1.0-beta25 from http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz

ROOT=$1
FILENAME="matconvnet-1.0-beta25"
TARNAME="$FILENAME.tar.gz"
DIR_COMPILENN="$FILENAME/matlab"

if [ ! -f "$ROOT/extern/$TARNAME" ]; then
  echo "Downloading $FILENAME."
  wget -nd "http://www.vlfeat.org/matconvnet/download/$TARNAME" -P "$ROOT/desc_doap"
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


#if [ -d "$ROOT/desc_doap/$FILENAME" ]; then
#  echo "Compiling $FILENAME"
#  cd -P "$ROOT/desc_doap/$FILENAME"
#  make ARCH=glnxa64 MEX=$MATLAB_HOME/mex
#  echo "Compilation done."
#fi

if [ -d "$ROOT/desc_doap" ]; then
  echo "Setup matlab and mex."
  cd -P "$ROOT/desc_doap"
  matlab -r "setup_matlab_once('$DIR_COMPILENN'); exit(0)";
  echo "Matlab setup done."
fi

if [ -f "$ROOT/desc_doap/$TARNAME" ]; then
  echo "Removing $TARNAME."
  rm "$ROOT/desc_doap/$TARNAME"
  echo "Removal done."
fi
