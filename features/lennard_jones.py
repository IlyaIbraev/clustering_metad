import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd

from utils import line_generator


def prepare_matrix(lj_filename: str, frames_filename: str) -> None:

    frames = set()

    with open(
        frames_filename, "r"
    ) as file:
        lines = file.readlines()
    for line in lines[1:]:
        frames_in_line = map(int, line.split())
        for frame in frames_in_line:
            frames.add(frame)

    # prepare collective variables dictionary
    cvs = {}
    with open(lj_filename, "r") as file:
        file_gen = line_generator(file)
        for line in file_gen:
            splitted = line.strip().split()
            if len(splitted) >= 3:
                if splitted[2] == "legend":
                    index = int(splitted[1][1:]) + 1
                    res1 = int(splitted[3].replace("\"", "").split(
                        "-")[1].split("_")[-1])-1
                    res2 = int(splitted[3].replace("\"", "").split(
                        "-")[2].split("_")[-1])-1
                    cvs[f"{res1}_{res2}"] = index
            if splitted[0][0] not in ("#", "@"):
                break

    # prepare LJ-interractions dataframe
    frame_number = 0
    lj_interract = pd.DataFrame({cv: [] for cv in cvs.keys()})
    with open(lj_filename, "r") as file:
        file_gen = line_generator(file)
        for line in file_gen:
            splitted = line.strip().split()
            if splitted[0][0] not in ("#", "@"):
                if frame_number in frames:
                    new_row = pd.Series(
                        {cv: float(splitted[cvs[cv]]) for cv in cvs.keys()})
                    lj_interract = pd.concat(
                        [lj_interract, new_row.to_frame().T],
                        ignore_index=True
                    )
                frame_number += 1

    rmsd = cdist(
        lj_interract, lj_interract
    ).astype(np.float32)

    rmsd.tofile("calculations/lennard_jones/rmsd.dat")
