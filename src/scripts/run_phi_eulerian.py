from typing import Optional, Tuple
from pathlib import Path

from src.fluid import flow_statistics as fstat
from src.myio.myio import MyPath


def main(
    parties_data_dir: MyPath,
    output_dir: MyPath,
    min_file_index: Optional[int],
    max_file_index: Optional[int],
    compute: Tuple[bool],
):
    output_dir = Path(output_dir) / "fluid"
    mean_phi_h5: Path = output_dir / "mean_phi.h5"

    if compute[0]:
        fstat.process_mean_phi(
            parties_data_dir=parties_data_dir,
            output_h5=mean_phi_h5,
            compute_err=True,
            min_file_index=min_file_index,
            max_file_index=max_file_index,
        )
