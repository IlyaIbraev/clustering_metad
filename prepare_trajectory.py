import os
from utils import read_colvar


def prepare_trajectory(topology: str, trajectory: str, weights: str, weights_field: str, num_frames: int):

    weights_data = read_colvar(weights)

    left, right = 0.0, 1e5
    mid = (left + right) / 2

    EPS = 1e-5

    while left + EPS < right:
        mid = (left + right) / 2
        if weights_data[weights_data[weights_field] > mid].shape[0] >= num_frames:
            left = mid
        else:
            right = mid

    # dump frames to frames.ndx

    with open("calculations/trajectory/frames.ndx", "w") as f:
        f.write("[ frames ]\n")
        for i, frame in enumerate(weights_data[weights_data[weights_field] > mid].index):
            if i % 100 == 0:
                f.write(f"\n")
            f.write(f"{frame} ")
        f.write("\n")

    # run GMX TRJCONV to extract frames

    os.system(
        f"gmx trjconv -f {trajectory} -s {topology} -fr calculations/trajectory/frames.ndx -o calculations/trajectory/frames.xtc << EOF\n"
        "1\n"
        "EOF\n"
    )
