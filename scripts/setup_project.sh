#!/bin/bash
ROOT=$(pwd)

# TILDE
echo "Starting setup for TILDE..."
echo "==========================="
$ROOT/scripts/det_tilde/setup_docker_image_for_tilde.sh $ROOT
echo "======================"
echo "Setup for TILDE: done."

# DOAP
echo "Start setup for DOAP..."
echo "======================="
$ROOT/scripts/desc_doap/setup_vlfeat.sh $ROOT
$ROOT/scripts/desc_doap/setup_matconvnet.sh $ROOT
echo "====================="
echo "Setup for DOAP: done."

# TFEAT
echo "Start setup for TFeat..."
echo "========================"
$ROOT/scripts/desc_tfeat/setup_tfeat.sh $ROOT
echo "======================"
echo "Setup for Tfeat: done."

# SuperPoint
echo "Start setup for SuperPoint..."
echo "========================"
$ROOT/scripts/pipe_superpoint/setup_superpoint.sh $ROOT
echo "======================"
echo "Setup for SuperPoint: done."

# LIFT
echo "Start setup for LIFT..."
echo "========================"
$ROOT/scripts/pipe_lift/setup_lift.sh $ROOT
echo "======================"
echo "Setup for LIFT: done."

# SIFT
echo "Start setup for SIFT..."
echo "========================"
$ROOT/scripts/pipe_sift/setup_sift.sh $ROOT
echo "======================"
echo "Setup for SIFT: done."
