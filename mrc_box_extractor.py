#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRC Box Extractor

This script extracts a 2D box from an MRC file centered around a specified pixel location.
It handles edge cases and provides error handling and logging capabilities.
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import mrcfile


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, logs to console only.

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("mrc_box_extractor")
    logger.setLevel(getattr(logging, log_level))

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def extract_box(
    mrc_path: str,
    center_x: int,
    center_y: int,
    box_size_x: int,
    box_size_y: int,
    logger: logging.Logger,
) -> np.ndarray:
    """Extract a 2D box from an MRC file.

    Args:
        mrc_path: Path to the MRC file
        center_x: X-coordinate of the box center
        center_y: Y-coordinate of the box center
        box_size_x: Width of the box
        box_size_y: Height of the box
        logger: Logger object

    Returns:
        2D numpy array containing the extracted box
    """
    logger.info(f"Opening MRC file: {mrc_path}")
    try:
        with mrcfile.open(mrc_path, mode="r", permissive=True) as mrc:
            # Verify that this is a 2D image
            if len(mrc.data.shape) != 2:
                raise ValueError(
                    f"Expected 2D image, but got {len(mrc.data.shape)}D data with shape {mrc.data.shape}"
                )

            image_height, image_width = mrc.data.shape
            logger.debug(f"Image dimensions: {image_width} x {image_height}")

            # Calculate box boundaries
            half_box_x = box_size_x // 2
            half_box_y = box_size_y // 2

            start_x = center_x - half_box_x
            start_y = center_y - half_box_y
            end_x = center_x + half_box_x + (box_size_x % 2)  # Handle odd box sizes
            end_y = center_y + half_box_y + (box_size_y % 2)

            logger.debug(f"Box coordinates: ({start_x}, {start_y}) to ({end_x}, {end_y})")

            # Check if box is completely outside the image
            if (
                start_x >= image_width
                or end_x <= 0
                or start_y >= image_height
                or end_y <= 0
            ):
                raise ValueError(
                    f"Box is completely outside the image boundaries ({image_width} x {image_height})"
                )

            # Handle partial boxes (on edges)
            padded_box = np.zeros((box_size_y, box_size_x), dtype=mrc.data.dtype)

            # Calculate valid region in the original image
            valid_start_x = max(0, start_x)
            valid_end_x = min(image_width, end_x)
            valid_start_y = max(0, start_y)
            valid_end_y = min(image_height, end_y)

            # Calculate corresponding region in the output box
            box_start_x = valid_start_x - start_x
            box_end_x = box_start_x + (valid_end_x - valid_start_x)
            box_start_y = valid_start_y - start_y
            box_end_y = box_start_y + (valid_end_y - valid_start_y)

            # Extract data and place in the box
            padded_box[box_start_y:box_end_y, box_start_x:box_end_x] = mrc.data[
                valid_start_y:valid_end_y, valid_start_x:valid_end_x
            ]

            logger.info(
                f"Successfully extracted box of size {box_size_x} x {box_size_y} "
                f"centered at ({center_x}, {center_y})"
            )
            return padded_box

    except FileNotFoundError:
        logger.error(f"MRC file not found: {mrc_path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied when accessing: {mrc_path}")
        raise
    except Exception as e:
        logger.error(f"Error extracting box from MRC file: {str(e)}")
        raise


def save_box(
    box: np.ndarray, output_path: str, logger: logging.Logger
) -> None:
    """Save the extracted box as an MRC file.

    Args:
        box: 2D numpy array containing the box data
        output_path: Path to save the output MRC file
        logger: Logger object
    """
    try:
        logger.info(f"Saving extracted box to: {output_path}")
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(box.astype(np.float32))  # MRC files typically use float32
        logger.info("Box saved successfully")
    except Exception as e:
        logger.error(f"Error saving box to MRC file: {str(e)}")
        raise


def parse_arguments():
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Extract a 2D box from an MRC file centered around a specified pixel."
    )
    parser.add_argument("mrc_path", help="Path to the input MRC file")
    parser.add_argument(
        "-x", "--center-x", type=int, required=True, help="X-coordinate of the box center"
    )
    parser.add_argument(
        "-y", "--center-y", type=int, required=True, help="Y-coordinate of the box center"
    )
    parser.add_argument(
        "--box-size-x", type=int, required=True, help="Width of the box"
    )
    parser.add_argument(
        "--box-size-y", type=int, required=True, help="Height of the box"
    )
    parser.add_argument(
        "-o", "--output", help="Path to save the output MRC file. If not provided, it will be derived from the input path."
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--log-file", help="Path to log file. If not provided, logs to console only."
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the extracted box using matplotlib"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Set up logging
    logger = setup_logging(args.log_level, args.log_file)

    try:
        # Extract box
        box = extract_box(
            args.mrc_path,
            args.center_x,
            args.center_y,
            args.box_size_x,
            args.box_size_y,
            logger,
        )

        # Determine output path if not provided
        if not args.output:
            base_name = os.path.splitext(os.path.basename(args.mrc_path))[0]
            args.output = f"{base_name}_box_x{args.center_x}_y{args.center_y}_w{args.box_size_x}_h{args.box_size_y}.mrc"
            logger.info(f"Output path not provided, using: {args.output}")

        # Save box
        save_box(box, args.output, logger)

        # Display box if requested
        if args.show:
            try:
                import matplotlib.pyplot as plt

                logger.info("Displaying extracted box")
                plt.figure(figsize=(8, 8))
                plt.imshow(box, cmap="gray")
                plt.title(
                    f"Box centered at ({args.center_x}, {args.center_y}), "
                    f"size: {args.box_size_x} x {args.box_size_y}"
                )
                plt.colorbar()
                plt.show()
            except ImportError:
                logger.warning("Matplotlib not installed. Cannot display the box.")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())