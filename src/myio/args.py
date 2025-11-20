# flocparties/io/args.py
"""Argument parsing for postprocessing scripts."""
import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="PARTIES_postprocess",
        description="Todo.",
    )

    parser.add_argument("-i", "--inp_dir", type=Path, help="Directory with .inp files.")
    parser.add_argument("-d", "--data_dir", type=Path, help="Directory with raw data.")
    parser.add_argument("-f", "--floc_dir", type=Path, help="Directory with floc files.")
    parser.add_argument("-m", "--metadata_file", type=Path, help="Path to metadata.ini.")
    parser.add_argument("-o", "--out_dir", type=Path, help="Directory for output.")
    parser.add_argument("-trn", action="store_true", help="Use trn subdirectory files.")

    args: argparse.Namespace = parser.parse_args()

    args.data_dir = args.data_dir or args.inp_dir
    args.metadata_file = args.metadata_file or (args.out_dir / "metadata.ini")  # Assuming metadata may be in the analysis dir
    args.out_dir.mkdir(exist_ok=True)

    return args
