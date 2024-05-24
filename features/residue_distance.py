import os
import numpy as np
import scipy
import mdtraj as md


from utils import read_colvar


def prepare_matrix(topology_filename: str, trajectory_filename: str) -> None:

    heavy_atoms_per_res = {}

    protein = md.load(topology_filename)
    for resid in range(protein.n_residues):
        heavy_atoms_per_res[resid] = protein.top.select(
            f"resid {resid} and mass >= 12"
        )

    n_atoms_in_protein = protein.n_atoms
    plumed_filename = "calculations/residue_distance/plumed_contactmap.dat"

    write_arr = []

    write_arr.append(
        f"MOLINFO STRUCTURE={topology_filename}"
    )
    write_arr.append(
        f"WHOLEMOLECULES ENTITY0=1-{n_atoms_in_protein}"
    )

    write_arr.append("")

    for resid in heavy_atoms_per_res:
        write_arr.append(
            f"residue_{resid}: GROUP ATOMS={','.join(map(str, heavy_atoms_per_res[resid]+1))}"
        )

    write_arr.append("")

    values_to_output = []

    for a in range(0, protein.n_residues):
        for b in range(a+1, protein.n_residues):

            write_arr.append(
                f"dist_{a}_{b}: DISTANCES GROUPA=residue_{a} GROUPB=residue_{b} LOWEST NOPBC"
            )

            values_to_output.append(f"dist_{a}_{b}.lowest")

    write_arr.append(
        f"PRINT ARG={','.join(values_to_output)} FILE=calculations/residue_distance/CONTACTMAP_COLVAR STRIDE=1"
    )

    with open(plumed_filename, "w") as file:
        file.write("\n".join(write_arr))

    # calculate distances using plumed driver
    os.system(
        f"plumed driver --mf_xtc {trajectory_filename} --plumed {plumed_filename}"
    )

    # prepare precalculated distance rmsd matrix
    distances = read_colvar("calculations/residue_distance/CONTACTMAP_COLVAR")
    del distances["time"]

    rmsd = scipy.spatial.distance.cdist(
        distances, distances).astype(np.float32)

    rmsd.tofile("calculations/residue_distance/rmsd.dat")
