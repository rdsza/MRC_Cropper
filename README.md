# MRC_Cropper

Python command line script that reads a 2D MRC file, extracts a box around a specified pixel index, and handles errors appropriately.

### This script:

- Uses the mrcfile library to read 2D MRC files (commonly used in cryo-EM)
- Implements comprehensive error handling with customizable logging
- Handles edge cases where the box might extend beyond image boundaries by padding with zeros
- Supports saving the extracted box as a new MRC file
- Provides an option to visualize the extracted box using matplotlib
- Includes type hints for better code documentation
- Follows PEP 8 style guidelines for Python code

### Usage :

python mrc_box_extractor.py path/to/image.mrc -x 100 -y 150 --box-size-x 64 --box-size-y 64 -o output.mrc --log-level DEBUG