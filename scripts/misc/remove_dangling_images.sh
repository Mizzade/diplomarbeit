#!/bin/bash
# Removes all danging and untagged docker images.
# @INPUT:
#   -
# @OUTPUT
#   -

docker rmi $(docker images -f "dangling=true" -q)
