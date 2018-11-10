#!/usr/bin/env python3
import sys
import subprocess
import typing
import os
import json

def get_setName_fileName_extension(file_path: str) -> (str, str, str):
    base_path, extension = os.path.splitext(file_path)
    set_name, file_name = base_path.split(os.sep)[-2:]

    return base_path, set_name, file_name, extension

if __name__ == '__main__':
    argv = sys.argv[1:]

    if len(argv) <= 0:
        raise RuntimeError("Missing argument <output_dir> and <file_list>. Abort")

    assert isinstance(argv[0], str)
    assert isinstance(argv[1], str)
    assert isinstance(json.loads(argv[1]), list)

    output_dir = argv[0]
    file_list = json.loads(argv[1])

    for file in file_list:
        # Build parameters for use_tilde.cpp
        base_path, set_name, file_name, extension = get_setName_fileName_extension(file)

        # Build parameters for use_tilde.cpp
        imageDir = (os.sep).join(base_path.split('/')[:-1])
        outputDir = os.path.join(output_dir, set_name)
        fileName = file_name + extension
        filterPath = '/home/tilde/TILDE/c++/Lib/filters'
        filterName = 'Mexico.txt'

        # Create directories otherwise use_tilde.cpp cannot save output.
        _keypoint_path = os.path.join(outputDir, 'keypoints')
        _heatmap_path = os.path.join(outputDir, 'scores')
        if not os.path.exists(_keypoint_path):
            os.makedirs(_keypoint_path, exist_ok=True)

        if not os.path.exists(_heatmap_path):
            os.makedirs(_heatmap_path, exist_ok=True)

        # Call use_tilde.cpp
        subprocess.check_call([
            './use_tilde',
            '--imageDir', imageDir,
            '--outputDir', outputDir,
            '--fileName', fileName,
            '--filterPath', filterPath,
            '--filterName', filterName])
