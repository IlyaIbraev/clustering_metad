import os
import numpy as np
from gromacs.fileformats.xpm import XPM


def prepare_matrix(topology_filename: str, trajectory_filename: str) -> None:

    os.system(
        f"gmx rms -f {trajectory_filename} -s {topology_filename} -o calculations/trajectory/rmsd.xvg -m calculations/trajectory/rmsd.xpm << EOF\n"
        "4\n"
        "4\n"
        "EOF\n"
    )

    rmsd_matrix = XPM(
        "calculations/trajectory/rmsd.xpm"
    ).array.astype(np.float32)

    rmsd_matrix.tofile("calculations/heavy_atoms/rmsd.dat")
