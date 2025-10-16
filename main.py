import argparse
from typing import Union, Optional
from pathlib import Path

from src import scripts
from src import globals

if __name__ == "__main__":

    output_dir: Path = Path("./output")
    parties_data_dir: Path = Path("./data")
    utexas_data_dir: Path = Path("./data")

    if globals.on_anvil:
        output_dir = Path("/home/x-lwidmer/Documents/PARTIES_postprocess/output")
        utexas_data_dir = Path("/home/x-lwidmer/Documents/PARTIES_postprocess/data")
        parties_data_dir = Path("/anvil/scratch/x-lwidmer/RUN9")

    parser = argparse.ArgumentParser(
        prog="PARTIES_postprocess",
        description="Run postprocessing scripts on output data of PARTIES",
        epilog="to change which postprocessing functions are run modify root level main.py",
    )
    parser.add_argument(
        "-pd",
        "--parties_data_dir",
        nargs="?",
        type=Path,
        default=parties_data_dir,
        help="directory in which PARTIES output data is located",
    )
    parser.add_argument(
        "-ud",
        "--utexas_data_dir",
        nargs="?",
        type=Path,
        default=utexas_data_dir,
        help="directory in which utexas data is located",
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        nargs="?",
        type=Path,
        default=output_dir,
        help="directory in which output_data should be stored",
    )
    parser.add_argument(
        "-min",
        "--min_file_index",
        nargs="?",
        type=int,
        default=None,
        help="minimal particle/fluid datafile index to be read",
    )
    parser.add_argument(
        "-max",
        "--max_file_index",
        nargs="?",
        type=int,
        default=None,
        help="maximal particle/fluid datafile index to be read",
    )
    parser.add_argument(
        "-trn",
        action="store_true",
        help="Wether to use Particle_XXX.h5 files in the trn subdirectory",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        nargs="?",
        type=int,
        default=None,
        help="Number of workers to use when multithreading / mutiprocessing (None for concurrent operation)",
    )
    parser.add_argument(
        "-thr",
        "--use_threading",
        action="store_true",
        help="Wether to use ThreadPoolExecutor instead of ProcessPoolExecutor",
    )
    args = parser.parse_args()

    scripts.run_floc_analysis.main(
        args.parties_data_dir,
        args.output_dir,
        args.trn,
        args.min_file_index,
        args.max_file_index,
        args.num_workers,
    )
    scripts.run_statist_steady_analysis.main(
        args.parties_data_dir,
        args.output_dir,
        args.min_file_index,
        args.max_file_index,
    )
    scripts.run_fluid_wall_analysis.main(
        args.parties_data_dir,
        args.utexas_data_dir,
        args.output_dir,
        args.min_file_index,
        args.max_file_index,
    )
