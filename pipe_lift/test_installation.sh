#!/bin/bash
# Runs LIFT pipeline for one image.
# @INPUT:
#   $1: absolute path of the project's root directory.
#   $2: Name of the image inside inputs folder WITHOUT extension.
#   $3: Only the extension of the image inside the inputs folder.
# @OUTPUT
#   -
ROOT=$1
IMAGE_NAME=$2
EXT=$3


LIFT_DIR="$ROOT/pipe_lift"
LIFT_TF_DIR="$LIFT_DIR/tf-lift"

INPUT_DIR='inputs'
OUTPUT_DIR='outputs'


if [ -d $LIFT_TF_DIR ]; then
  echo "Starting LIFT pipeline."

  if [ ! -d "$LIFT_DIR/$INPUT_DIR" ]; then
    echo "Creating input dir."
    cd $LIFT_DIR
    mkdir $INPUT_DIR
    # cp "$ROOT/data/v_set_04/1.png" "$INPUT_DIR/1.png"
    # magick mogrify -resize 15% -quality 100% 1.png
    #convert -rotate "90" "$INPUT_DIR/1.png" "$INPUT_DIR/COCO_test2014_000000000016.jpg"
  fi

  if [ ! -d "$LIFT_DIR/$OUTPUT_DIR" ]; then
    echo "Creating output dir."
    cd $LIFT_DIR
    mkdir $OUTPUT_DIR
  fi

  cd $LIFT_TF_DIR

  echo "Generating Keypoints"
  echo "===================="
  python main.py \
    --task=test \
    --subtask=kp \
    --logdir=../pretrained_models/release-aug \
    --test_img_file=../inputs/${IMAGE_NAME}.${EXT} \
    --test_out_file=../outputs/${IMAGE_NAME}_kp_aug.txt \
    --use_batch_norm=False \
    --mean_std_type=hardcoded

  echo "Generating Orientation"
  echo "======================"
  python main.py \
    --task=test \
    --subtask=ori \
    --logdir=../pretrained_models/release-aug \
    --test_img_file=../inputs/${IMAGE_NAME}.${EXT} \
    --test_out_file=../outputs/${IMAGE_NAME}_ori_aug.txt \
    --test_kp_file=../outputs/${IMAGE_NAME}_kp_aug.txt \
    --use_batch_norm=False \
    --mean_std_type=hardcoded

  echo "Generating Descriptors"
  echo "======================"
  python main.py \
    --task=test \
    --subtask=desc \
    --logdir=../pretrained_models/release-aug \
    --test_img_file=../inputs/${IMAGE_NAME}.${EXT} \
    --test_out_file=../outputs/${IMAGE_NAME}_desc_aug.h5 \
    --test_kp_file=../outputs/${IMAGE_NAME}_ori_aug.txt \
    --use_batch_norm=False \
    --mean_std_type=hardcoded

  echo "LIFT pipeline successfull."
fi
